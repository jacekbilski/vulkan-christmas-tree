# Vulkan-christmas-tree

This repo is purely "for fun", it's about messing around with drawing something in 3D using Vulkan. It more or less follows what happened in [rusted-christmas-tree](https://github.com/jacekbilski/rusted-christmas-tree) and [wasm-christmas-tree](https://github.com/jacekbilski/wasm-christmas-tree) repositories, just using Vulkan instead of OpenGL or WebGL. Rust language also stays.

## Vulkan

This thing is _way_ more complex than OpenGL or WebGL. It is much more low-level, gives far more power, but also requires much more knowledge about how the things actually work under the hood. Long story short:

## Validation layers

They help understand Vulkan and avoid mistakes when using it. To enable them, just add `"validation-layers"` to a default features in `Cargo.toml`.

## Some basic ideas

First of all, Vulkan is asynchronous (well, OpenGL and WebGL also were, but not that much in your face). What I basically will do all the time is to prepare commands that will be doing something interesting, like copying data or executing shaders, and submit them to queues for execution. The tricky part is, I have no guarantees when or in what order those commans will be executed. When a specific order is required, fences and semaphores come into play.

## Higher-level ideas

What does it take to draw a single frame on the screen.

Well, quite a lot. It consists of two parts: producing the final image and the presenting it on the screen. Presenting is easy, merely one command, but it requires putting together a few things. Producing the image to present is what is really interesting and complex.

Producing the image (drawing in short) is done in a render pass. This pass contains one or more subpasses. Each subpass can be responsible for some other aspect of rendering the scene, like geometry, shadows, ray-tracing, etc. They mostly define what their output will look like, for example save color result to an image, use 4 samples per pixel and ignore produced depth data.

Next are the pipelines. They again combine a lot of things, but the most important thing is that they put together shader modules and actual memory buffers together.

## Low-level gory details

### Initialization

To start I need an instance. I can also declare here what extensions do I require, like "VK_KHR_surface".

From an instance I can get a physical device. It's usually a graphics card, but could also be a compute accelerator or even a software emulator.

From a physical device I can get queue families. Queues are used to tell the GPU to do something. Using different queues enables parallelism (think "threads"). Families - I do not understand them yet.

Having selected one queue family that supports graphics I can get a device. It's one of the core concepts in Vulkan and used almost everywhere. It allows communication with the GPU. Keep it close at all times.

### Drawing something to the memory

This is where things start to get messy. In order to draw something, I need a graphics pipeline.

It basically binds together shaders, data those shaders will be operating on, and a render pass.

A render pass consists attachments and passes. It basically defines the format of the image that will be drawn, a bit how to draw. It can get very complex.

Framebuffer is the realization of a render pass. Render pass alone just defines things, framebuffer makes that all happen.

Dynamic state - it keeps things that might change over time, for example: viewport. It will be necessary to support for example window resizing. But there are more things I don't yet understand.

And finally drawing. It is done by issuing a command to a queue.

Here, it starts the render pass, then draws something using the pipeline, finishes the render pass and copies what the GPU created to a buffer. The buffer can now be transformed into an image and saved to the disk. I didn't draw anything on the screen yet!

### Drawing something on the screen

First, obvious, thing is that I need a window with a VK_Surface. `winit` can deal with that. It also provides me with a list of all required extensions. The window should also be open until I close it explicitly.

Next thing is a swapchain. It's a series of images the app should be drawing into. Think double/tripple buffering. This is also where the resolution of the thing I'll be drawing is chosen. That also means that every window resizing requires recreating the whole swapchain.

Mow magical things happen. I need a loop now that repeats itself every frame drawn. There are two parts to that. First is waiting for a `winit::event::Event::RedrawEventsCleared`. That means all my draw commans were executed. But that's not enough. If I'm issuing commands faster than the images are displayed, I'll not be able to get the next image from a swapchain, because it will still be waiting to be presented. Therefore, I have those `sync_objects`.

Inside a loop I need to:

1. wait for the fences (I don't know yet what they are)
1. acquire next image from the swapchain,
1. submit a command buffer with all render passes and `draw` commands for execution,
1. present the results,
1. recreate the swapchain if necessary.

### Uniform buffers

I need two things:

1. buffer itself,
1. and a descriptor.

Next, I create a descriptor. First, I take a layout from the pipeline, id needs to match that in the shaders (`binding`). Then I create the descriptor from the layout and add the buffer to it.

At last, I pass the descriptor to the `draw` command.

### Instancing

Not that complex. First of all, I need to create a struct holding the data per instance.

Second, I need to create an instances buffer holding actual instances information. Technically it's a yet another buffer, like vertex buffer.

Third, creating the pipeline changes a bit. In `vertex_input` call I need to say, that I'll be passing two buffers, one for vertices, one for instances.

Fourth, to the `draw` call I'm passing in instances buffer as a second vertex buffer.

As a last step I need to adapt the shader. Simply adding more `layout(location = x) in vec3 y;` is enough, just make sure locations are correct that is `x` is greater by one than last location taken from vertex buffer. 

## Tree model

The tree model is taken from Internet: https://www.turbosquid.com/3d-models/forest-pack-obj-free/917762
