#include "locus/pipeline/stage.hpp"

#include <unistd.h>

#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace locus::pipeline {

PipelineStage::PipelineStage(const model::LlamaModel& model,
                             std::uint32_t layer_begin,
                             std::uint32_t layer_end,
                             std::uint32_t n_blocks)
    : model_(model),
      layer_begin_(layer_begin),
      layer_end_(layer_end),
      n_layers_(model.hparams().n_layers),
      n_embd_(model.hparams().n_embd),
      n_vocab_(model.hparams().n_vocab),
      cache_(model.make_cache(n_blocks)),
      ws_(model.make_workspace()) {
    if (layer_begin_ >= layer_end_ || layer_end_ > n_layers_) {
        throw std::invalid_argument("PipelineStage: bad layer range");
    }
}

Message PipelineStage::step(const Message& in) {
    const bool last = is_last();
    const std::uint32_t before = seq_.n_tokens;
    // Every stage processes the same tokens in order, so the entry's
    // position must match this stage's own next slot; a mismatch means
    // the stages fell out of lockstep.
    if (in.position != before) {
        throw std::invalid_argument(
            "PipelineStage: position out of lockstep");
    }
    if (!cache_.ensure_capacity(seq_, 1)) {
        throw std::runtime_error("PipelineStage: cache exhausted");
    }

    std::vector<float> out(last ? n_vocab_ : n_embd_);
    if (is_first()) {
        if (in.type != MsgType::kToken) {
            throw std::invalid_argument(
                "PipelineStage: first stage expects a kToken message");
        }
        model_.forward_layers(in.token, {}, layer_begin_, layer_end_,
                              cache_, seq_, ws_, out);
    } else {
        if (in.type != MsgType::kActivation) {
            throw std::invalid_argument(
                "PipelineStage: expects a kActivation message");
        }
        if (in.data.size() != n_embd_) {
            throw std::invalid_argument(
                "PipelineStage: activation size mismatch");
        }
        model_.forward_layers(0, in.data, layer_begin_, layer_end_,
                              cache_, seq_, ws_, out);
    }

    // forward_layers advances the sequence only on the final stage
    // (layer_end == n_layers); a non-final stage advances its own
    // position so the next token's KV lands at the right slot.
    if (!last) {
        seq_.n_tokens = before + 1;
    }

    if (last) {
        return make_logits(in.request_id, in.position, std::move(out));
    }
    return make_activation(in.request_id, in.position, std::move(out));
}

bool PipelineStage::run(int in_fd, int out_fd) {
    bool ok = true;
    for (;;) {
        Message in;
        const ReadResult r = read_message(in_fd, in);
        if (r == ReadResult::kEof) {
            break;  // clean shutdown
        }
        if (r != ReadResult::kOk) {
            ok = false;
            break;
        }
        Message out;
        try {
            out = step(in);
        } catch (const std::exception&) {
            ok = false;
            break;
        }
        if (!write_message(out_fd, out)) {
            ok = false;
            break;
        }
    }
    // Close downstream first so the next stage sees EOF and the chain
    // drains, then the upstream fd.
    ::close(out_fd);
    ::close(in_fd);
    return ok;
}

}  // namespace locus::pipeline
