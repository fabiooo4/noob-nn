use std::io;
use std::io::Write;

use noob_nn::{NeuralNetworkOptions, NeuralNetwork};
use rand::random_range;

const EPOCHS: usize = 10000;

fn main() {
    // Clear the screen
    print!("\x1b[2J\x1b[H");

    let mut predictor = NeuralNetwork::new(2, 3, 1, NeuralNetworkOptions::default());
    let old_predictor = predictor.clone();

    // Training
    let data = xor_truthtable();
    for epoch in 0..=EPOCHS {
        let (input, expected) = data[random_range(0..data.len())];

        predictor.train(&input, &[expected]);

        draw_xor(&predictor, 20);
        println!(
            "Training epoch: {epoch}/{EPOCHS} {:.0}%",
            (epoch as f32 / EPOCHS as f32) * 100.
        )
    }

    println!("\nBefore training:\n{old_predictor:#?}");
    println!("\nAfter training:\n{predictor:#?}\n");

    println!("Predicted values:");
    for (input, _) in data {
        let output = predictor.feed_forward(&input);
        println!("{input:?} -> {output:?}");
    }
}

const fn xor_truthtable() -> [([f32; 2], f32); 4] {
    [
        ([0., 0.], 0.),
        ([0., 1.], 1.),
        ([1., 0.], 1.),
        ([1., 1.], 0.),
    ]
}

fn draw_xor(nn: &NeuralNetwork, size: usize) {
    let block = "█";

    // Batch output in a buffer
    let mut out = io::BufWriter::new(io::stdout().lock());

    // Hide and move the cursor to top left
    write!(out, "\x1b[?25l\x1b[H").unwrap();

    // Top labels
    if size > 10 {
        writeln!(out, "(0,1){}(1,1)", " ".repeat(size * 2 - 10)).unwrap();
    }

    for y in (0..size).rev() {
        for x in 0..size {
            // A terminal cell is taller than it is wide so each x is drawn twice to compensate
            for _ in 0..2 {
                let val = nn.feed_forward(&[x as f32 / size as f32, y as f32 / size as f32])[0];
                let color_val = (val * 255.0).clamp(0.0, 255.0) as u8;

                write!(out, "\x1b[38;2;{c};{c};{c}m{block}", c = color_val).unwrap();
            }
        }
        writeln!(out).unwrap();
    }

    // Reset formatting
    write!(out, "\x1b[0m").unwrap();

    // Bottom labels
    if size > 10 {
        writeln!(out, "(0,0){}(1,0)", " ".repeat(size * 2 - 10)).unwrap();
    }

    // Show the cursor again
    write!(out, "\x1b[?25h").unwrap();

    out.flush().unwrap();
}
