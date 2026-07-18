#include "cppllm/backend/vulkan/context.hpp"

#include <stdexcept>

#if defined(CPPLLM_HAS_VULKAN_KERNELS)

#include <vulkan/vulkan.h>

#include <cstring>
#include <vector>

#include "matvec_spv.h"

namespace cppllm::backend::vk {

namespace {

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) {
        throw std::runtime_error(std::string("vulkan: ") + what);
    }
}

}  // namespace

struct VulkanContext::Impl {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t queue_family = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;

    struct HostBuffer {
        VkBuffer buf = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        void* map = nullptr;
    };

    void init() {
        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "cppllm";
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

        VkDescriptorSetLayoutBinding binds[3]{};
        for (std::uint32_t i = 0; i < 3; ++i) {
            binds[i].binding = i;
            binds[i].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dli{};
        dli.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dli.bindingCount = 3;
        dli.pBindings = binds;
        check(vkCreateDescriptorSetLayout(device, &dli, nullptr,
                                          &dsl),
              "create dsl");

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.size = 2 * sizeof(std::uint32_t);
        VkPipelineLayoutCreateInfo pli{};
        pli.sType =
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &dsl;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pcr;
        check(vkCreatePipelineLayout(device, &pli, nullptr,
                                     &layout),
              "create pipeline layout");

        VkShaderModuleCreateInfo smi{};
        smi.sType =
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smi.codeSize = sizeof(cppllm_matvec_spv);
        smi.pCode = cppllm_matvec_spv;
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
        cpci.layout = layout;
        check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1,
                                       &cpci, nullptr, &pipeline),
              "create pipeline");
        vkDestroyShaderModule(device, sm, nullptr);

        VkDescriptorPoolSize dps{};
        dps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        dps.descriptorCount = 3;
        VkDescriptorPoolCreateInfo dpi{};
        dpi.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpi.maxSets = 1;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &dps;
        check(vkCreateDescriptorPool(device, &dpi, nullptr,
                                     &dpool),
              "create descriptor pool");
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
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyPipelineLayout(device, layout, nullptr);
            vkDestroyDescriptorSetLayout(device, dsl, nullptr);
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

void VulkanContext::matvec_f32(Buffer w, std::uint32_t rows,
                               std::uint32_t cols, Buffer x,
                               Buffer out) {
    Impl& im = *impl_;
    auto& wb = *static_cast<Impl::HostBuffer*>(w.impl);
    auto& xb = *static_cast<Impl::HostBuffer*>(x.impl);
    auto& ob = *static_cast<Impl::HostBuffer*>(out.impl);

    VkDescriptorSetAllocateInfo dsa{};
    dsa.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsa.descriptorPool = im.dpool;
    dsa.descriptorSetCount = 1;
    dsa.pSetLayouts = &im.dsl;
    VkDescriptorSet ds;
    check(vkAllocateDescriptorSets(im.device, &dsa, &ds),
          "allocate descriptor set");

    VkDescriptorBufferInfo infos[3] = {
        {wb.buf, 0, VK_WHOLE_SIZE},
        {xb.buf, 0, VK_WHOLE_SIZE},
        {ob.buf, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet writes[3]{};
    for (std::uint32_t i = 0; i < 3; ++i) {
        writes[i].sType =
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = ds;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(im.device, 3, writes, 0, nullptr);

    VkCommandBufferAllocateInfo cba{};
    cba.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cba.commandPool = im.pool;
    cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cba.commandBufferCount = 1;
    VkCommandBuffer cmd;
    check(vkAllocateCommandBuffers(im.device, &cba, &cmd),
          "allocate command buffer");

    VkCommandBufferBeginInfo cbb{};
    cbb.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbb.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(cmd, &cbb), "begin cmd");
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      im.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            im.layout, 0, 1, &ds, 0, nullptr);
    const std::uint32_t pc[2] = {rows, cols};
    vkCmdPushConstants(cmd, im.layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(pc), pc);
    vkCmdDispatch(cmd, (rows + 63) / 64, 1, 1);
    check(vkEndCommandBuffer(cmd), "end cmd");

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    check(vkQueueSubmit(im.queue, 1, &si, VK_NULL_HANDLE),
          "submit");
    check(vkQueueWaitIdle(im.queue), "wait idle");

    vkFreeCommandBuffers(im.device, im.pool, 1, &cmd);
    check(vkResetDescriptorPool(im.device, im.dpool, 0),
          "reset descriptor pool");
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

}  // namespace cppllm::backend::vk

#else  // !CPPLLM_HAS_VULKAN_KERNELS

namespace cppllm::backend::vk {

struct VulkanContext::Impl {};

bool VulkanContext::available() { return false; }

VulkanContext::VulkanContext() {
    throw std::runtime_error(
        "cppllm built without Vulkan kernels");
}

VulkanContext::~VulkanContext() = default;

namespace {

[[noreturn]] void no_kernels() {
    throw std::runtime_error(
        "cppllm built without Vulkan kernels");
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

}  // namespace cppllm::backend::vk

#endif
