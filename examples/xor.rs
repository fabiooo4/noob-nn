use std::io;
use std::io::Write;

use noob_nn::{NetworkOptions, NeuralNetwork};
use rand::random_range;

const EPOCHS: usize = 50000;

fn main() {
    // Clear the screen
    print!("\x1b[2J\x1b[H");

    let mut classifier = NeuralNetwork::new(2, 3, 1, NetworkOptions::default());
    let old_classifier = classifier.clone();

    // Training
    let data = xor_truthtable();
    for epoch in 0..EPOCHS {
        let (input, expected) = data[random_range(0..data.len())];

        classifier.train(&input, &[expected]);

        draw_xor(&classifier, 20);
        println!(
            "Training epoch: {epoch}/{EPOCHS} {:.0}%",
            (epoch as f32 / EPOCHS as f32) * 100.
        )
    }

    println!("\nBefore training:\n{old_classifier:#?}");
    println!("\nAfter training:\n{classifier:#?}\n");

    let input = [0., 1.];
    let output = classifier.feed_forward(&input);

    println!("{input:?} -> {output:?}");
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

    // \x1b[?25l : Hide the cursor so it doesn't bounce around the screen
    // \x1b[H    : Move cursor to home. We REMOVED \x1b[2J (Clear Screen) to
    //             prevent the blank-screen flicker. We will just overwrite the old frame.
    write!(out, "\x1b[?25l\x1b[H").unwrap();

    // Top labels
    if size > 10 {
        writeln!(out, "(0,1){}(1,1)", " ".repeat(size * 2 - 10)).unwrap();
    }

    for y in 0..size {
        for x in 0..size {
            // A terminal cell is taller than it is wide so each x is drawn twice to compensate
            for _ in 0..2 {
                let val = nn.feed_forward(&[x as f32 / size as f32, y as f32 / size as f32])[0];
                let color_val = (val * 255.0).clamp(0.0, 255.0) as u8;

                // Write to the memory buffer instead of the terminal
                write!(out, "\x1b[38;2;{v};{v};{v}m{block}", v = color_val).unwrap();
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

    // Flush pushes the fully assembled frame to the terminal in a single operation
    out.flush().unwrap();
}
