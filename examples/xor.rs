use noob_nn::NeuralNetwork;

fn main() {
    let classifier = NeuralNetwork::new(2, 2, 1);

    println!("{classifier:#?}");

    let input = [0., 1.];
    let output = classifier.feed_forward(&input);

    // TODO: training

    println!("{input:?} -> {output:?}");
}
