#import <tensorflow/core/public/session.h>

#import "ViewController.h"

// I took these examples from the test set. To make this an app that is useful
// in practice, you would need to add code that records audio and then extracts
// the acoustic properties from the audio.
static float maleExample[] = {
	0.174272105181833,0.0694110453235828,0.190874106652007,0.115601979109401,
	0.228279274326553,0.112677295217152,4.48503835015822,61.7649083141473,
	0.950972024663856,0.635199178449614,0.0500274876305662,0.174272105181833,
	0.102045991451092,0.0183276059564719,0.246153846153846,1.62129934210526,
	0.0078125,7,6.9921875,0.209310986964618,
};

static float femaleExample[] = {
	0.19753980448103,0.0347746034366121,0.198683834048641,0.182660944206009,
	0.218712446351931,0.0360515021459227,2.45939730314823,9.56744874023233,
	0.839523285188244,0.226976814502006,0.185064377682403,0.19753980448103,
	0.173636160583901,0.0470127326150832,0.271186440677966,1.61474609375,
	0.2109375,15.234375,15.0234375,0.0389615584623385,
};

@implementation ViewController
{
	tensorflow::GraphDef graph;
	tensorflow::Session *session;
}

- (void)viewDidLoad {
	[super viewDidLoad];

	NSString *path = [[NSBundle mainBundle] pathForResource:@"inference" ofType:@"pb"];

	if ([self loadGraphFromPath:path] && [self createSession]) {
		[self predict:maleExample];
		[self predict:femaleExample];
		session->Close();
	}
}

- (BOOL)loadGraphFromPath:(NSString *)path
{
	auto status = ReadBinaryProto(tensorflow::Env::Default(), path.fileSystemRepresentation, &graph);
	if (!status.ok()) {
		NSLog(@"Error reading graph: %s", status.error_message().c_str());
		return NO;
	}

	// This prints out the names of the nodes in the graph.
	auto nodeCount = graph.node_size();
	NSLog(@"Node count: %d", nodeCount);
	for (auto i = 0; i < nodeCount; ++i) {
		auto node = graph.node(i);
		NSLog(@"Node %d: %s '%s'", i, node.op().c_str(), node.name().c_str());
	}

	return YES;
}

- (BOOL)createSession
{
	tensorflow::SessionOptions options;
	auto status = tensorflow::NewSession(options, &session);
	if (!status.ok()) {
		NSLog(@"Error creating session: %s", status.error_message().c_str());
		return NO;
	}

	status = session->Create(graph);
	if (!status.ok()) {
		NSLog(@"Error adding graph to session: %s", status.error_message().c_str());
		return NO;
	}

	return YES;
}

- (void)predict:(float *)example {
	// Define the tensor for the input data. This tensor takes one example
	// at a time, and the example has 20 features.
	tensorflow::Tensor x(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, 20 }));

	// Put the input data into the tensor.
	auto input = x.tensor<float, 2>();
	for (int i = 0; i < 20; ++i) {
		input(0, i) = example[i];
	}

	// The feed dictionary for doing inference.
	std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
        {"inputs/x-input", x}
    };

	// We want to run these nodes.
	std::vector<std::string> nodes = {
        {"model/y_pred"},
        {"inference/inference"}
    };

	// The results of running the nodes are stored in this vector.
	std::vector<tensorflow::Tensor> outputs;

	// Run the session.
	auto status = session->Run(inputs, nodes, {}, &outputs);
	if (!status.ok()) {
		NSLog(@"Error running model: %s", status.error_message().c_str());
		return;
	}

	// Print out the result of the prediction.
	auto y_pred = outputs[0].tensor<float, 2>();
	NSLog(@"Probability spoken by a male: %f%%", y_pred(0, 0));

	auto isMale = outputs[1].tensor<float, 2>();
	if (isMale(0, 0)) {
		NSLog(@"Prediction: male");
	} else {
		NSLog(@"Prediction: female");
	}
}

@end
