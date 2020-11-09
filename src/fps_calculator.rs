use std::collections::VecDeque;
use std::time::Instant;

const FPS_ARRAY_SIZE: usize = 100;

pub struct FpsCalculator {
    frame_times: VecDeque<Instant>,
}

impl FpsCalculator {
    pub(crate) fn new() -> Self {
        let mut frame_times: VecDeque<Instant> = VecDeque::with_capacity(FPS_ARRAY_SIZE);
        frame_times.push_back(Instant::now());
        FpsCalculator { frame_times }
    }

    pub(crate) fn tick(&mut self) {
        let earliest_frame: Instant = if self.frame_times.len() == FPS_ARRAY_SIZE {
            self.frame_times.pop_front().unwrap()
        } else {
            *(self.frame_times.front().unwrap())
        };
        let elapsed = earliest_frame.elapsed();
        let _fps = 1_000_000.0 * self.frame_times.len() as f64 / elapsed.as_micros() as f64;
        // println!("FPS: {:?}, elapsed: {:?}", _fps, elapsed);
        self.frame_times.push_back(Instant::now());
    }

    pub(crate) fn last_frame_time_secs(&self) -> f32 {
        if self.frame_times.len() < 2 {
            return 0.0;
        }
        let last = self.frame_times[self.frame_times.len() - 1];
        let second_last = self.frame_times[self.frame_times.len() - 2];
        let duration = last.duration_since(second_last);
        duration.as_secs_f32() + duration.subsec_micros() as f32 / 1_000_000.0
    }
}
