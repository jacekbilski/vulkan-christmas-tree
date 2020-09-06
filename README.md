# Vulkan-christmas-tree

This repo is purely "for fun", it's about messing around with drawing something in 3D using Vulkan. It more or less follows what happened in [rusted-christmas-tree](https://github.com/jacekbilski/rusted-christmas-tree) and [wasm-christmas-tree](https://github.com/jacekbilski/wasm-christmas-tree) repositories, just using Vulkan instead of OpenGL or WebGL. Rust language also stays.

## Vulkan

This thing is _way_ more complex than OpenGL or WebGL. It is much more low-level, gives far more power, but also requires much more knowledge about how the things actually work under the hood. Long story short:

### Initialization

To start I need an instance. I can also declare here what extensions do I require, like "VK_KHR_surface".

From an instance I can get a physical device. It's usually a graphics card, but could also be a compute accelerator or even a software emulator.

From a physical device I can get queue families. Queues are used to tell the GPU to do something. Using different queues enables parallelism (think "threads"). Families - I do not understand them yet.

Having selected one queue family that supports graphics I can get a device. It's one of the core concepts in Vulkan and used almost everywhere. It allows communication with the GPU. Keep it close at all times.

### Drawing something to the memory

This is where things start to get messy. In order to draw something, I need a graphics pipeline.

Graphics pipeline might look like this:

```rust
let pipeline = Arc::new(GraphicsPipeline::start()
    .vertex_input_single_buffer::<Vertex>()
    .vertex_shader(vs.main_entry_point(), ())
    .viewports_dynamic_scissors_irrelevant(1)
    .fragment_shader(fs.main_entry_point(), ())
    .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
    .build(device.clone())
    .unwrap());
```

It basically binds together shaders, data those shaders will be operating on, and a render pass.

A render pass consists attachments and passes. It basically defines the format of the image that will be drawn, a bit how to draw. It can get very complex.

Framebuffer is the realization of a render pass. Render pass alone just defines things, framebuffer makes that all happen.

Dynamic state - it keeps things that might change over time, for example: viewport. It will be necessary to support for example window resizing. But ther are more things I don't yet understand.

And finally drawing. It is done by issuing a command to a queue, like so:

```rust
let mut command_builder = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap();
command_builder
    .begin_render_pass(framebuffer.clone(), false, vec![[0.015_7, 0., 0.360_7, 1.0].into()])
    .unwrap()

    .draw(pipeline.clone(), &dynamic_state, vertex_buffer.clone(), (), ())
    .unwrap()

    .end_render_pass()
    .unwrap()

    .copy_image_to_buffer(image.clone(), buf.clone())
    .unwrap();
let command_buffer = command_builder.build().unwrap();
```

Here, it starts the render pass, then draws something using the pipeline, finishes the render pass and copies what the GPU created to a buffer. The buffer can now be transformed into an image and saved to the disk. I didn't draw anything on the screen yet!

### Drawing something on the screen

First, obvious, thing is that I need a window with a VK_Surface. `winit` can deal with that. It also provides me with a list of all required extensions. The window should also be open until I close it explicitly.

Next thing is a swapchain. It's a series of images the app should be drawing into. Think double/tripple buffering. This is also where the resolution of the thing I'll be drawing is chosen. That also means that every window resizing requires recreating the whole swapchain.

Mow magical things happen. I need a loop now that repeats itself every frame drawn. There are two parts to that. First is waiting for a `winit::event::Event::RedrawEventsCleared`. That means all my draw commans were executed. But that's not enough. If I'm issuing commands faster than the images are displayed, I'll not be able to get the next image from a swapchain, because it will still be waiting to be presented. So there's this `previous_frame_end`, a `Future` I need to wait on.

Inside a loop I need to:

1. acquire next image from the swapchain,
1. build a command buffer with all render passes and `draw` commands,
1. actually wait until `previous_frame_end` and `acquire_future` resolve (aka wait until previous frame ... something, not clear to me just yet, basically wait until we're ready to start drawing),
1. execute command buffer,
1. present the results.
