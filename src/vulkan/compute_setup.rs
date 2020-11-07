use crate::vulkan::core::VulkanCore;

pub struct VulkanComputeSetup {
    core: VulkanCore,
}

impl VulkanComputeSetup {
    pub fn new(core: VulkanCore) -> Self {
        VulkanComputeSetup { core }
    }
}
