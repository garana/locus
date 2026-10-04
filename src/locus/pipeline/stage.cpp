#include "locus/pipeline/stage.hpp"

#include <unistd.h>

#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace locus::pipeline {

PipelineStage::PipelineStage(const model::TransformerModel& model,
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

Message PipelineStage::step(const Message& in) { return step(seq_, in); }

Message PipelineStage::step(kv::PagedKvCache::Seq& seq,
                            const Message& in) {
    const bool last = is_last();
    const std::uint32_t before = seq.n_tokens;
    // Every stage processes the same tokens in order, so the entry's
    // position must match this session's own next slot; a mismatch means
    // the stages fell out of lockstep.
    if (in.position != before) {
        throw std::invalid_argument(
            "PipelineStage: position out of lockstep");
    }
    if (!cache_.ensure_capacity(seq, 1)) {
        throw std::runtime_error("PipelineStage: cache exhausted");
    }

    std::vector<float> out(last ? n_vocab_ : n_embd_);
    if (is_first()) {
        if (in.type != MsgType::kToken) {
            throw std::invalid_argument(
                "PipelineStage: first stage expects a kToken message");
        }
        model_.forward_layers(in.token, {}, layer_begin_, layer_end_,
                              cache_, seq, ws_, out);
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
                              cache_, seq, ws_, out);
    }

    // forward_layers advances the sequence only on the final stage
    // (layer_end == n_layers); a non-final stage advances its own
    // position so the next token's KV lands at the right slot.
    if (!last) {
        seq.n_tokens = before + 1;
    }

    if (last) {
        return make_logits(in.request_id, in.position, std::move(out));
    }
    return make_activation(in.request_id, in.position, std::move(out));
}

void PipelineStage::step_batch(std::span<const BatchInput> ins,
                               std::vector<BatchOutput>& outs) {
    outs.assign(ins.size(), BatchOutput{});
    const bool first = is_first();
    const bool last = is_last();

    // Validate and reserve capacity per entry, collecting the ones that
    // pass into a dense batch. A bad or out-of-lockstep frame, or a cache
    // exhaustion, fails only its own slot; the rest of the batch runs.
    std::vector<std::uint32_t> idx;  // original index of each batch member
    std::vector<kv::PagedKvCache::Seq*> seqs;
    std::vector<std::uint32_t> before;  // pre-step n_tokens per member
    std::vector<tok::TokenId> toks;     // first stage only
    std::vector<float> hidden;          // n*n_embd, non-first stage only
    idx.reserve(ins.size());
    seqs.reserve(ins.size());
    before.reserve(ins.size());

    for (std::uint32_t i = 0; i < ins.size(); ++i) {
        const Message& in = *ins[i].in;
        kv::PagedKvCache::Seq& seq = *ins[i].seq;
        try {
            // Same per-session lockstep rule as step(): the entry's
            // position must equal this session's own next slot.
            if (in.position != seq.n_tokens) {
                throw std::invalid_argument(
                    "PipelineStage: position out of lockstep");
            }
            if (first) {
                if (in.type != MsgType::kToken) {
                    throw std::invalid_argument(
                        "PipelineStage: first stage expects a kToken "
                        "message");
                }
                // Range-check the token HERE, per entry, so a bad id
                // fails only its own slot. forward_batch_layers validates
                // the range by throwing for the whole call, which would
                // otherwise fail every co-batched session -- and token is
                // an unchecked int32 off the wire, so any upstream can
                // send one.
                if (in.token < 0 ||
                    static_cast<std::uint32_t>(in.token) >= n_vocab_) {
                    throw std::invalid_argument(
                        "PipelineStage: token id out of vocab");
                }
            } else {
                if (in.type != MsgType::kActivation) {
                    throw std::invalid_argument(
                        "PipelineStage: expects a kActivation message");
                }
                if (in.data.size() != n_embd_) {
                    throw std::invalid_argument(
                        "PipelineStage: activation size mismatch");
                }
            }
            if (!cache_.ensure_capacity(seq, 1)) {
                throw std::runtime_error("PipelineStage: cache exhausted");
            }
        } catch (const std::exception& e) {
            outs[i].ok = false;
            outs[i].err = e.what();
            continue;
        }
        idx.push_back(i);
        seqs.push_back(&seq);
        before.push_back(seq.n_tokens);  // == in.position, unchanged above
        if (first) {
            toks.push_back(in.token);
        } else {
            hidden.insert(hidden.end(), in.data.begin(), in.data.end());
        }
    }

    const std::uint32_t n = static_cast<std::uint32_t>(idx.size());
    if (n == 0) {
        return;  // every entry already failed validation
    }

    const std::uint32_t width = last ? n_vocab_ : n_embd_;
    std::vector<float> out(static_cast<std::size_t>(n) * width);
    model_.forward_batch_layers(
        first ? std::span<const tok::TokenId>(toks)
              : std::span<const tok::TokenId>(),
        first ? std::span<const float>()
              : std::span<const float>(hidden),
        layer_begin_, layer_end_, cache_, seqs, ws_, out);

    // forward_batch_layers advances each seq only on the final stage; a
    // non-final stage bumps its own position so the next token's KV lands
    // at the right slot (lockstep), exactly as step() does.
    if (!last) {
        for (std::uint32_t t = 0; t < n; ++t) {
            seqs[t]->n_tokens = before[t] + 1;
        }
    }

    for (std::uint32_t t = 0; t < n; ++t) {
        const std::uint32_t i = idx[t];
        const Message& in = *ins[i].in;
        std::vector<float> slice(
            out.begin() + static_cast<std::size_t>(t) * width,
            out.begin() + static_cast<std::size_t>(t + 1) * width);
        outs[i].out =
            last ? make_logits(in.request_id, in.position, std::move(slice))
                 : make_activation(in.request_id, in.position,
                                   std::move(slice));
    }
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

void PipelineStage::reset() { reset(seq_); }

void PipelineStage::reset(kv::PagedKvCache::Seq& seq) {
    cache_.release(seq);
    seq = kv::PagedKvCache::Seq{};
}

}  // namespace locus::pipeline
