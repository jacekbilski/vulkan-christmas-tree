#![windows_subsystem = "windows"]

use std::f32::consts::{FRAC_PI_8, TAU};
use std::thread;
use std::time::{Duration, Instant};

use vulkan::Vulkan;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::ElementState::Pressed;
use winit::event::{Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::fps_calculator::FpsCalculator;
use crate::scene::Scene;

mod vulkan;

mod color_mesh;
mod coords;
mod fps_calculator;
mod scene;
mod textured_mesh;

const AUTO_ROTATION_SPEED_RAD_PER_SEC: f32 = TAU / 30.0;

const MAX_FPS: u8 = 60;

const APPLICATION_NAME: &'static str = "Vulkan Christmas Tree";

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = init_window(&event_loop);
    let mut vulkan = Vulkan::new(&window, APPLICATION_NAME);
    let scene = Scene::setup(&mut vulkan, &window);
    main_loop(vulkan, window, scene, event_loop);
}

fn init_window(event_loop: &EventLoop<()>) -> winit::window::Window {
    let window = winit::window::WindowBuilder::new()
        .with_title(APPLICATION_NAME)
        .with_inner_size(PhysicalSize::new(1, 1))
        .build(event_loop)
        .expect("Failed to create window.");
    let monitor = window
        .current_monitor()
        .or(window.primary_monitor())
        .or(window.available_monitors().next());
    let screen_size = monitor
        .map(|monitor| monitor.size())
        .map(|size| PhysicalSize::new(size.width / 2, size.height / 2))
        .unwrap_or(PhysicalSize::new(1600, 900));
    window.request_inner_size(screen_size).unwrap();

    window
}

fn main_loop(
    mut vulkan: Vulkan,
    window: winit::window::Window,
    mut scene: Scene,
    event_loop: EventLoop<()>,
) {
    let mut fps_calculator = FpsCalculator::new();
    let mut autorotate = false;
    let mut mouse_rotating = false;
    let mut last_cursor_position: PhysicalPosition<f64> = PhysicalPosition::new(0.0, 0.0);
    let desired_frame_duration = Duration::from_secs_f32(1.0 / MAX_FPS as f32);
    event_loop.run(move |event, elwt| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            vulkan.wait_device_idle();
            elwt.exit();
        }
        Event::WindowEvent {
            event:
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(code),
                            state: Pressed,
                            ..
                        },
                    ..
                },
            ..
        } => match code {
            KeyCode::Escape => {
                vulkan.wait_device_idle();
                elwt.exit();
            }
            KeyCode::ArrowUp => {
                autorotate = false;
                let angle_change = FRAC_PI_8 / 4.;
                scene.rotate_camera_vertically(angle_change, &mut vulkan);
            }
            KeyCode::ArrowDown => {
                autorotate = false;
                let angle_change = FRAC_PI_8 / 4.;
                scene.rotate_camera_vertically(-angle_change, &mut vulkan);
            }
            KeyCode::ArrowLeft => {
                autorotate = false;
                let angle_change = FRAC_PI_8 / 4.;
                scene.rotate_camera_horizontally(-angle_change, &mut vulkan);
            }
            KeyCode::ArrowRight => {
                autorotate = false;
                let angle_change = FRAC_PI_8 / 4.;
                scene.rotate_camera_horizontally(angle_change, &mut vulkan);
            }
            KeyCode::KeyR => {
                autorotate = !autorotate;
            }
            _ => (),
        },
        Event::WindowEvent {
            event:
                WindowEvent::MouseInput {
                    button: MouseButton::Left,
                    state,
                    ..
                },
            ..
        } => {
            mouse_rotating = state == Pressed;
        }
        Event::WindowEvent {
            event:
                WindowEvent::MouseWheel {
                    delta: MouseScrollDelta::LineDelta(_, vertical),
                    ..
                },
            ..
        } => {
            scene.change_camera_distance(-0.5 * vertical, &mut vulkan);
        }
        Event::WindowEvent {
            event: WindowEvent::CursorMoved { position, .. },
            ..
        } => {
            if mouse_rotating {
                let x_diff = position.x - last_cursor_position.x;
                let y_diff = position.y - last_cursor_position.y;

                let angle_change = FRAC_PI_8 / 128.;
                scene.rotate_camera_horizontally(-angle_change * x_diff as f32, &mut vulkan);
                scene.rotate_camera_vertically(angle_change * y_diff as f32, &mut vulkan);
            }
            last_cursor_position = position;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(new_size),
            ..
        } => {
            vulkan.wait_device_idle();
            scene.framebuffer_resized(new_size, &mut vulkan);
            vulkan.framebuffer_resized(new_size.width, new_size.height);
        }
        Event::AboutToWait => {
            window.request_redraw();
        }
        WindowEvent::RedrawRequested => {
            let frame_start = Instant::now();
            fps_calculator.tick();
            let last_frame_time_secs = fps_calculator.last_frame_time_secs();
            if autorotate {
                scene.rotate_camera_horizontally(
                    AUTO_ROTATION_SPEED_RAD_PER_SEC * last_frame_time_secs,
                    &mut vulkan,
                );
            }
            vulkan.draw_frame(last_frame_time_secs);
            let frame_end = Instant::now();
            let actual_frame_duration = frame_end - frame_start;
            if actual_frame_duration < desired_frame_duration {
                thread::sleep(desired_frame_duration - actual_frame_duration);
            }
        }
        Event::LoopExiting => {
            vulkan.wait_device_idle();
        }
        _ => (),
    });
}
