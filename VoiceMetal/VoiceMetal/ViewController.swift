import UIKit
import Metal
import MetalPerformanceShaders

// I took these examples from the test set. To make this an app that is useful
// in practice, you would need to add code that records audio and then extracts
// the acoustic properties from the audio.

let maleExample: [Float] = [
	0.174272105181833,0.0694110453235828,0.190874106652007,0.115601979109401,
	0.228279274326553,0.112677295217152,4.48503835015822,61.7649083141473,
	0.950972024663856,0.635199178449614,0.0500274876305662,0.174272105181833,
	0.102045991451092,0.0183276059564719,0.246153846153846,1.62129934210526,
	0.0078125,7,6.9921875,0.209310986964618,
]

let femaleExample: [Float] = [
	0.19753980448103,0.0347746034366121,0.198683834048641,0.182660944206009,
	0.218712446351931,0.0360515021459227,2.45939730314823,9.56744874023233,
	0.839523285188244,0.226976814502006,0.185064377682403,0.19753980448103,
	0.173636160583901,0.0470127326150832,0.271186440677966,1.61474609375,
	0.2109375,15.234375,15.0234375,0.0389615584623385,
]

class ViewController: UIViewController {

  var device: MTLDevice!
  var commandQueue: MTLCommandQueue!

  var layer: MPSCNNFullyConnected!
  var inputImage: MPSImage!
  var outputImage: MPSImage!

  override func viewDidLoad() {
    super.viewDidLoad()

    createGraph()
    predict(example: maleExample)
    predict(example: femaleExample)
  }

  func createGraph() {
    device = MTLCreateSystemDefaultDevice()
    guard device != nil else {
      fatalError("Error: This device does not support Metal")
    }

    guard MPSSupportsMTLDevice(device) else {
      fatalError("Error: This device does not support Metal Performance Shaders")
    }

    commandQueue = device.makeCommandQueue()

    // Load the weights and bias value.
    let W_url = Bundle.main.url(forResource: "W", withExtension: "bin")
    let b_url = Bundle.main.url(forResource: "b", withExtension: "bin")
    let W_data = try! Data(contentsOf: W_url!)
    let b_data = try! Data(contentsOf: b_url!)

    // The logistic regression is computed by the formula sigmoid((W * x) + b).
    // That's exactly the same computation that a fully-connected layer performs.

    let sigmoid = MPSCNNNeuronSigmoid(device: device)

    let layerDesc = MPSCNNConvolutionDescriptor(kernelWidth: 1, kernelHeight: 1, inputFeatureChannels: 20, outputFeatureChannels: 1, neuronFilter: sigmoid)

    // Create the fully-connected layer.
    W_data.withUnsafeBytes { W in
      b_data.withUnsafeBytes { b in
        layer = MPSCNNFullyConnected(device: device, convolutionDescriptor: layerDesc, kernelWeights: W, biasTerms: b, flags: .none)
      }
    }

    let inputImgDesc = MPSImageDescriptor(channelFormat: .float16, width: 1, height: 1, featureChannels: 20)
    let outputImgDesc = MPSImageDescriptor(channelFormat: .float16, width: 1, height: 1, featureChannels: 1)

    // The input and output are placed in MPSImage objects.
    inputImage = MPSImage(device: device, imageDescriptor: inputImgDesc)
    outputImage = MPSImage(device: device, imageDescriptor: outputImgDesc)
  }

  func convert(example: [Float], to image: MPSImage) {
    // The input data must be placed into the MTLTexture from the MPSImage.
    // First we convert it to 16-bit floating point, then copy these floats
    // into the texture slices. (Note: this will fail if the number of channels
    // is not a multiple of 4 -- in that case you should add padding bytes to 
    // the example array.)
    var example = example
    let input16 = float32to16(&example, count: example.count)
    input16.withUnsafeBufferPointer { ptr in
      for i in 0..<inputImage.texture.arrayLength {
        let region = MTLRegion(origin: MTLOriginMake(0, 0, 0), size: MTLSizeMake(1, 1, 1))
        inputImage.texture.replace(region: region, mipmapLevel: 0, slice: i, withBytes: ptr.baseAddress!.advanced(by: i*4), bytesPerRow: MemoryLayout<Float16>.stride * 4, bytesPerImage: 0)
      }
    }
  }

  func predict(example: [Float]) {
    // Load the example data into an MPSImage so we can use it on the GPU.
    convert(example: example, to: inputImage)

    // Perform the computations on the GPU.
    let commandBuffer = commandQueue.makeCommandBuffer()
    layer.encode(commandBuffer: commandBuffer, sourceImage: inputImage, destinationImage: outputImage)
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    // Print out the result of the prediction.
    let y_pred = outputImage.toFloatArray()
    print("Probability spoken by a male: \(y_pred[0])%")

    if y_pred[0] > 0.5 {
      print("Prediction: male")
    } else {
      print("Prediction: female")
    }
  }
}
