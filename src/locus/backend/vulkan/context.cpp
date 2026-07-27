#include "locus/backend/vulkan/context.hpp"

#include <stdexcept>

#if defined(LOCUS_HAS_VULKAN_KERNELS)

#include <vulkan/vulkan.h>

#include <bit>
#include <cstring>
#include <vector>

#include "attn_mla_spv.h"
#include "attn_paged_spv.h"
#include "matvec_f16_spv.h"
#include "matvec_q4_0_spv.h"
#include "matvec_q5_0_spv.h"
#include "matvec_q4_k_spv.h"
#include "matvec_q5_k_spv.h"
#include "matvec_q6_k_spv.h"
#include "matvec_q2_k_spv.h"
#include "matvec_iq2_xxs_spv.h"
#include "matvec_q8_0_spv.h"
#include "matvec_spv.h"
#include "matvec_t_spv.h"
#include "rmsnorm_spv.h"
#include "rope_spv.h"
#include "silu_mul_spv.h"

namespace locus::backend::vk {

namespace {

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) {
        throw std::runtime_error(std::string("vulkan: ") + what);
    }
}

/** Static description of each kernel's interface. */
struct KernelDesc {
    const std::uint32_t* spv;
    std::size_t spv_bytes;
    std::uint32_t n_buffers;
    std::uint32_t push_bytes;
};

const KernelDesc& kernel_desc(Kernel k) {
    static const KernelDesc descs[] = {
        {locus_matvec_spv, sizeof(locus_matvec_spv), 3, 28},
        {locus_matvec_f16_spv, sizeof(locus_matvec_f16_spv), 3,
         28},
        {locus_matvec_q8_0_spv, sizeof(locus_matvec_q8_0_spv),
         3, 28},
        {locus_matvec_q4_0_spv, sizeof(locus_matvec_q4_0_spv),
         3, 28},
        {locus_matvec_q5_0_spv, sizeof(locus_matvec_q5_0_spv),
         3, 28},
        {locus_matvec_q4_k_spv, sizeof(locus_matvec_q4_k_spv),
         3, 28},
        {locus_matvec_q5_k_spv, sizeof(locus_matvec_q5_k_spv),
         3, 28},
        {locus_matvec_q6_k_spv, sizeof(locus_matvec_q6_k_spv),
         3, 28},
        {locus_matvec_q2_k_spv, sizeof(locus_matvec_q2_k_spv),
         3, 28},
        {locus_matvec_iq2_xxs_spv,
         sizeof(locus_matvec_iq2_xxs_spv), 4, 28},
        {locus_matvec_t_spv, sizeof(locus_matvec_t_spv), 3,
         20},
        {locus_rmsnorm_spv, sizeof(locus_rmsnorm_spv), 3, 12},
        {locus_rope_spv, sizeof(locus_rope_spv), 2, 36},
        {locus_silu_mul_spv, sizeof(locus_silu_mul_spv), 3, 4},
        {locus_attn_paged_spv, sizeof(locus_attn_paged_spv), 5,
         36},
        {locus_attn_mla_spv, sizeof(locus_attn_mla_spv), 6,
         36},
    };
    return descs[static_cast<int>(k)];
}

}  // namespace

struct VulkanContext::Impl {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t queue_family = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;

    struct Pipe {
        VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
    };
    Pipe pipes[static_cast<int>(Kernel::kCount_)];

    /** Open batch state (begin_batch .. end_batch). */
    VkCommandBuffer batch_cmd = VK_NULL_HANDLE;

    struct HostBuffer {
        VkBuffer buf = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        void* map = nullptr;
    };

    void init() {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "locus";
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &app;
#if defined(__APPLE__)
        const char* exts[] = {
            "VK_KHR_portability_enumeration"};
        ici.enabledExtensionCount = 1;
        ici.ppEnabledExtensionNames = exts;
        ici.flags =
            VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
        check(vkCreateInstance(&ici, nullptr, &instance),
              "create instance");

        std::uint32_t n = 0;
        vkEnumeratePhysicalDevices(instance, &n, nullptr);
        if (n == 0) {
            throw std::runtime_error("vulkan: no devices");
        }
        std::vector<VkPhysicalDevice> devs(n);
        vkEnumeratePhysicalDevices(instance, &n, devs.data());

        for (auto d : devs) {
            std::uint32_t qn = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(d, &qn,
                                                     nullptr);
            std::vector<VkQueueFamilyProperties> qs(qn);
            vkGetPhysicalDeviceQueueFamilyProperties(d, &qn,
                                                     qs.data());
            for (std::uint32_t i = 0; i < qn; ++i) {
                if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    phys = d;
                    queue_family = i;
                    break;
                }
            }
            if (phys != VK_NULL_HANDLE) {
                break;
            }
        }
        if (phys == VK_NULL_HANDLE) {
            throw std::runtime_error("vulkan: no compute queue");
        }

        const float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = queue_family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
#if defined(__APPLE__)
        const char* dexts[] = {"VK_KHR_portability_subset"};
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = dexts;
#endif
        check(vkCreateDevice(phys, &dci, nullptr, &device),
              "create device");
        vkGetDeviceQueue(device, queue_family, 0, &queue);

        VkCommandPoolCreateInfo cpi{};
        cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpi.queueFamilyIndex = queue_family;
        check(vkCreateCommandPool(device, &cpi, nullptr, &pool),
              "create command pool");

        for (int k = 0; k < static_cast<int>(Kernel::kCount_);
             ++k) {
            create_pipe(static_cast<Kernel>(k));
        }

        // Sized for the largest realistic per-token batch (a few
        // hundred dispatches); reset after every end_batch().
        VkDescriptorPoolSize dps{};
        dps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        dps.descriptorCount = 4096;
        VkDescriptorPoolCreateInfo dpi{};
        dpi.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpi.maxSets = 1024;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &dps;
        check(vkCreateDescriptorPool(device, &dpi, nullptr,
                                     &dpool),
              "create descriptor pool");
    }

    void create_pipe(Kernel k) {
        const KernelDesc& kd = kernel_desc(k);
        Pipe& p = pipes[static_cast<int>(k)];

        std::vector<VkDescriptorSetLayoutBinding> binds(
            kd.n_buffers);
        for (std::uint32_t i = 0; i < kd.n_buffers; ++i) {
            binds[i].binding = i;
            binds[i].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dli{};
        dli.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dli.bindingCount = kd.n_buffers;
        dli.pBindings = binds.data();
        check(vkCreateDescriptorSetLayout(device, &dli, nullptr,
                                          &p.dsl),
              "create dsl");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size = kd.push_bytes;
        VkPipelineLayoutCreateInfo pli{};
        pli.sType =
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &p.dsl;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
        check(vkCreatePipelineLayout(device, &pli, nullptr,
                                     &p.layout),
              "create pipeline layout");

        VkShaderModuleCreateInfo smi{};
        smi.sType =
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smi.codeSize = kd.spv_bytes;
        smi.pCode = kd.spv;
        VkShaderModule sm;
        check(vkCreateShaderModule(device, &smi, nullptr, &sm),
              "create shader module");

        VkComputePipelineCreateInfo cpci{};
        cpci.sType =
            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = sm;
        cpci.stage.pName = "main";
        cpci.layout = p.layout;
        check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1,
                                       &cpci, nullptr,
                                       &p.pipeline),
              "create pipeline");
        vkDestroyShaderModule(device, sm, nullptr);
    }

    HostBuffer make_buffer(VkDeviceSize size) {
        HostBuffer b;
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(device, &bci, nullptr, &b.buf),
              "create buffer");

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, b.buf, &req);
        VkPhysicalDeviceMemoryProperties props;
        vkGetPhysicalDeviceMemoryProperties(phys, &props);
        std::uint32_t type = ~0u;
        const VkMemoryPropertyFlags want =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (std::uint32_t i = 0;
             i < props.memoryTypeCount; ++i) {
            if ((req.memoryTypeBits & (1u << i)) &&
                (props.memoryTypes[i].propertyFlags & want) ==
                    want) {
                type = i;
                break;
            }
        }
        if (type == ~0u) {
            throw std::runtime_error(
                "vulkan: no host-visible memory");
        }
        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = type;
        check(vkAllocateMemory(device, &mai, nullptr, &b.mem),
              "allocate memory");
        check(vkBindBufferMemory(device, b.buf, b.mem, 0),
              "bind memory");
        check(vkMapMemory(device, b.mem, 0, size, 0, &b.map),
              "map memory");
        return b;
    }

    void destroy_buffer(HostBuffer& b) {
        if (b.mem != VK_NULL_HANDLE) {
            vkUnmapMemory(device, b.mem);
            vkFreeMemory(device, b.mem, nullptr);
        }
        if (b.buf != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, b.buf, nullptr);
        }
        b = {};
    }

    ~Impl() {
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
            vkDestroyDescriptorPool(device, dpool, nullptr);
            for (const Pipe& p : pipes) {
                vkDestroyPipeline(device, p.pipeline, nullptr);
                vkDestroyPipelineLayout(device, p.layout,
                                        nullptr);
                vkDestroyDescriptorSetLayout(device, p.dsl,
                                             nullptr);
            }
            vkDestroyCommandPool(device, pool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
        }
    }
};

bool VulkanContext::available() {
    try {
        VulkanContext ctx;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

VulkanContext::VulkanContext() : impl_(new Impl) { impl_->init(); }

VulkanContext::~VulkanContext() = default;

VulkanContext::Buffer VulkanContext::create_buffer(
    std::size_t bytes) {
    auto* hb = new Impl::HostBuffer(impl_->make_buffer(bytes));
    return Buffer{hb};
}

void VulkanContext::destroy_buffer(Buffer b) {
    auto* hb = static_cast<Impl::HostBuffer*>(b.impl);
    if (hb != nullptr) {
        impl_->destroy_buffer(*hb);
        delete hb;
    }
}

void VulkanContext::write_buffer(Buffer b,
                                 std::span<const std::byte> data) {
    auto* hb = static_cast<Impl::HostBuffer*>(b.impl);
    std::memcpy(hb->map, data.data(), data.size());
}

void VulkanContext::read_buffer(Buffer b,
                                std::span<std::byte> out) {
    auto* hb = static_cast<Impl::HostBuffer*>(b.impl);
    std::memcpy(out.data(), hb->map, out.size());
}

void* VulkanContext::mapped(Buffer b) {
    return static_cast<Impl::HostBuffer*>(b.impl)->map;
}

void VulkanContext::begin_batch() {
    Impl& im = *impl_;
    VkCommandBufferAllocateInfo cba{};
    cba.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cba.commandPool = im.pool;
    cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cba.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(im.device, &cba,
                                   &im.batch_cmd),
          "allocate command buffer");
    VkCommandBufferBeginInfo cbb{};
    cbb.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbb.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(im.batch_cmd, &cbb), "begin cmd");
}

void VulkanContext::dispatch(Kernel k,
                             std::span<const Buffer> buffers,
                             std::span<const std::uint32_t> push,
                             std::uint32_t groups_x) {
    Impl& im = *impl_;
    const Impl::Pipe& p = im.pipes[static_cast<int>(k)];
    const KernelDesc& kd = kernel_desc(k);
    if (buffers.size() != kd.n_buffers ||
        push.size() * 4 != kd.push_bytes) {
        throw std::runtime_error("vulkan: dispatch shape");
    }

    VkDescriptorSetAllocateInfo dsa{};
    dsa.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsa.descriptorPool = im.dpool;
    dsa.descriptorSetCount = 1;
    dsa.pSetLayouts = &p.dsl;
    VkDescriptorSet ds;
    check(vkAllocateDescriptorSets(im.device, &dsa, &ds),
          "allocate descriptor set");

    std::vector<VkDescriptorBufferInfo> infos(buffers.size());
    std::vector<VkWriteDescriptorSet> writes(buffers.size());
    for (std::size_t i = 0; i < buffers.size(); ++i) {
        auto* hb =
            static_cast<Impl::HostBuffer*>(buffers[i].impl);
        infos[i] = {hb->buf, 0, VK_WHOLE_SIZE};
        writes[i] = {};
        writes[i].sType =
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = ds;
        writes[i].dstBinding = static_cast<std::uint32_t>(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(
        im.device, static_cast<std::uint32_t>(writes.size()),
        writes.data(), 0, nullptr);

    vkCmdBindPipeline(im.batch_cmd,
                      VK_PIPELINE_BIND_POINT_COMPUTE,
                      p.pipeline);
    vkCmdBindDescriptorSets(
        im.batch_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout,
        0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(im.batch_cmd, p.layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       kd.push_bytes, push.data());
    vkCmdDispatch(im.batch_cmd, groups_x, 1, 1);

    // Serialize consecutive dispatches (writes feed reads).
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(im.batch_cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &mb, 0, nullptr, 0, nullptr);
}

void VulkanContext::end_batch() {
    Impl& im = *impl_;
    check(vkEndCommandBuffer(im.batch_cmd), "end cmd");
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &im.batch_cmd;
    check(vkQueueSubmit(im.queue, 1, &si, VK_NULL_HANDLE),
          "submit");
    check(vkQueueWaitIdle(im.queue), "wait idle");
    vkFreeCommandBuffers(im.device, im.pool, 1, &im.batch_cmd);
    im.batch_cmd = VK_NULL_HANDLE;
    check(vkResetDescriptorPool(im.device, im.dpool, 0),
          "reset descriptor pool");
}

void VulkanContext::copy_buffer(Buffer src, std::size_t src_off,
                                Buffer dst, std::size_t dst_off,
                                std::size_t bytes) {
    Impl& im = *impl_;
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(im.batch_cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                         &mb, 0, nullptr, 0, nullptr);
    VkBufferCopy region{src_off, dst_off, bytes};
    vkCmdCopyBuffer(im.batch_cmd,
                    static_cast<Impl::HostBuffer*>(src.impl)->buf,
                    static_cast<Impl::HostBuffer*>(dst.impl)->buf,
                    1, &region);
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(im.batch_cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &mb, 0, nullptr, 0, nullptr);
}

void VulkanContext::matvec_f32(Buffer w, std::uint32_t rows,
                               std::uint32_t cols, Buffer x,
                               Buffer out) {
    begin_batch();
    const Buffer bufs[] = {w, x, out};
    const std::uint32_t push[] = {
        rows, cols, 0, 0, 0, 0, std::bit_cast<std::uint32_t>(1.0f)};
    dispatch(Kernel::kMatvecF32, bufs, push, (rows + 63) / 64);
    end_batch();
}

void VulkanContext::matvec_f32(std::span<const float> w,
                               std::uint32_t rows,
                               std::uint32_t cols,
                               std::span<const float> x,
                               std::span<float> out) {
    if (w.size() != static_cast<std::size_t>(rows) * cols ||
        x.size() != cols || out.size() != rows) {
        throw std::runtime_error("vulkan: matvec size mismatch");
    }
    Buffer wb = create_buffer(w.size_bytes());
    Buffer xb = create_buffer(x.size_bytes());
    Buffer ob = create_buffer(out.size_bytes());
    write_buffer(wb, std::as_bytes(w));
    write_buffer(xb, std::as_bytes(x));
    matvec_f32(wb, rows, cols, xb, ob);
    read_buffer(ob, std::as_writable_bytes(out));
    destroy_buffer(wb);
    destroy_buffer(xb);
    destroy_buffer(ob);
}

}  // namespace locus::backend::vk

#else  // !LOCUS_HAS_VULKAN_KERNELS

namespace locus::backend::vk {

struct VulkanContext::Impl {};

bool VulkanContext::available() { return false; }

VulkanContext::VulkanContext() {
    throw std::runtime_error(
        "locus built without Vulkan kernels");
}

VulkanContext::~VulkanContext() = default;

namespace {

[[noreturn]] void no_kernels() {
    throw std::runtime_error(
        "locus built without Vulkan kernels");
}

}  // namespace

VulkanContext::Buffer VulkanContext::create_buffer(std::size_t) {
    no_kernels();
}
void VulkanContext::destroy_buffer(Buffer) { no_kernels(); }
void VulkanContext::write_buffer(Buffer,
                                 std::span<const std::byte>) {
    no_kernels();
}
void VulkanContext::read_buffer(Buffer, std::span<std::byte>) {
    no_kernels();
}
void* VulkanContext::mapped(Buffer) { no_kernels(); }
void VulkanContext::begin_batch() { no_kernels(); }
void VulkanContext::dispatch(Kernel, std::span<const Buffer>,
                             std::span<const std::uint32_t>,
                             std::uint32_t) {
    no_kernels();
}
void VulkanContext::end_batch() { no_kernels(); }
void VulkanContext::copy_buffer(Buffer, std::size_t, Buffer,
                                std::size_t, std::size_t) {
    no_kernels();
}
void VulkanContext::matvec_f32(Buffer, std::uint32_t,
                               std::uint32_t, Buffer, Buffer) {
    no_kernels();
}
void VulkanContext::matvec_f32(std::span<const float>,
                               std::uint32_t, std::uint32_t,
                               std::span<const float>,
                               std::span<float>) {
    no_kernels();
}

}  // namespace locus::backend::vk

#endif
