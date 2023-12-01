#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "utils.hpp"
#include <malloc.h>
#include <opencv2/opencv.hpp>
#include "httplib.h"

#define USE_FP32 //USE_FP16
#define INPUT_NAME "data"
#define OUTPUT_NAME "prob"
#define MAX_BATCH_SIZE 8

using namespace nvinfer1;
using namespace std;
static Logger gLogger;

float sigmoid(float x) {
    return (1 / (1 + exp(-x)));
}

static std::vector<BlockArgs>
	block_args_list = {
		BlockArgs{1, 3, 1, 1, 32, 16, 0.25, true},
		BlockArgs{2, 3, 2, 6, 16, 24, 0.25, true},
		BlockArgs{2, 5, 2, 6, 24, 40, 0.25, true},
		BlockArgs{3, 3, 2, 6, 40, 80, 0.25, true},
		BlockArgs{3, 5, 1, 6, 80, 112, 0.25, true},
		BlockArgs{4, 5, 2, 6, 112, 192, 0.25, true},
		BlockArgs{1, 3, 1, 6, 192, 320, 0.25, true}};

static std::map<std::string, GlobalParams>
	global_params_map = {
		// input_h,input_w,num_classes,batch_norm_epsilon,
		// width_coefficient,depth_coefficient,depth_divisor, min_depth
		{"b0", GlobalParams{224, 224, 1000, 0.001, 1.0, 1.0, 8, -1}},
		{"b1", GlobalParams{240, 240, 1000, 0.001, 1.0, 1.1, 8, -1}},
		{"b2", GlobalParams{260, 260, 1000, 0.001, 1.1, 1.2, 8, -1}},
		{"b3", GlobalParams{300, 300, 1000, 0.001, 1.2, 1.4, 8, -1}},
		{"b4", GlobalParams{380, 380, 1000, 0.001, 1.4, 1.8, 8, -1}},
		{"b5", GlobalParams{456, 456, 1000, 0.001, 1.6, 2.2, 8, -1}},
		{"b6", GlobalParams{528, 528, 1000, 0.001, 1.8, 2.6, 8, -1}},
		{"b7", GlobalParams{600, 600, 1000, 0.001, 2.0, 3.1, 8, -1}},
		{"b8", GlobalParams{672, 672, 1000, 0.001, 2.2, 3.6, 8, -1}},
		{"l2", GlobalParams{800, 800, 1000, 0.001, 4.3, 5.3, 8, -1}},
        {"b0-signal", GlobalParams{224, 224, 8, 0.001, 1.0, 1.0, 8, -1}},
};

std::vector<std::vector<float>> read_txt_to_mat(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件： " << filename << std::endl;
        exit(1);
    }

    std::vector<std::vector<float>> mat;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<float> row;
        std::string value;
        while (std::getline(iss, value, ',')) {
            row.push_back(std::stof(value));
        }
        mat.push_back(row);
    }

    file.close();
    return mat;
}

std::vector<float> prepareMatrix(const cv::Mat &img, const GlobalParams &global_params) {
    int c = 1;
    int h = global_params.input_h;
    int w = global_params.input_w;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);

    std::vector<float> mean_value{0.456};
    std::vector<float> std_value{0.224}; 
    std::vector<cv::Mat> input_channels;
    cv::split(resized, input_channels);

    for (auto i = 0; i < input_channels.size(); i++)
    {
        input_channels[0].convertTo(input_channels[0], CV_32FC1, 1.0 / (std_value[i]), (0.0 - mean_value[i]) / std_value[i]);
    }

    std::vector<float> result(h * w * c);
    auto data = result.data();
    int channel_length = h * w;
    for (int i = 0; i < c; ++i) {
        memcpy(data, input_channels[0].data, channel_length * sizeof(float));
        data += channel_length;  // 指针后移channel_length个单位
    }

    return result;
}

std::vector<float> prepareImage(const cv::Mat &img, const GlobalParams &global_params) {
    int c = 3;
    int h = global_params.input_h;
    int w = global_params.input_w;

    // std::cout << "Raw Image: " << img.size() <<'\n';
	// std::cout << "Preprocess Image: " << '[' << h << " x " << w << ']' <<'\n';

    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(w, h), 0, 0, cv::INTER_CUBIC);

    // cv::Mat img_float;
    // resized.convertTo(img_float, CV_32FC3, 1.f / 255.0);
    // std::vector<cv::Mat> input_channels(c);
    // cv::split(img_float, input_channels);

    std::vector<float> mean_value{ 0.485, 0.456, 0.406 };
    std::vector<float> std_value{ 0.229, 0.224, 0.225 }; 
    std::vector<cv::Mat> input_channels(c);
    cv::split(resized, input_channels);
 
    for (auto i = 0; i < input_channels.size(); i++)
    {
        input_channels[i].convertTo(input_channels[i], CV_32FC1, 1.0 / (std_value[i] * 255.0), (0.0 - mean_value[i]) / std_value[i]);
    }

    std::vector<float> result(h * w * c);
    auto data = result.data();
    int channel_length = h * w;
    for (int i = 0; i < c; ++i) {
        memcpy(data, input_channels[i].data, channel_length * sizeof(float));
        data += channel_length;  // 指针后移channel_length个单位
    }
    return result;
}

bool readFileToString(std::string file_name, std::string &fileData) {

    std::ifstream file(file_name.c_str(), std::ifstream::binary);

    if (file) {
        // Calculate the file's size, and allocate a buffer of that size.
        file.seekg(0, file.end);
        const int file_size = file.tellg();
        char *file_buf = new char[file_size + 1];
        //make sure the end tag \0 of string.
        memset(file_buf, 0, file_size + 1);

        // Read the entire file into the buffer.
        file.seekg(0, std::ios::beg);
        file.read(file_buf, file_size);

        if (file) {
            fileData.append(file_buf);
        } else {
            std::cout << "error: only " << file.gcount() << " could be read";
            fileData.append(file_buf);
            return false;
        }
        file.close();
        delete[]file_buf;
    } else {
        return false;
    }
    return true;
}

ICudaEngine *createEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt, std::string path_wts, std::vector<BlockArgs> block_args_list, GlobalParams global_params)
{
	float bn_eps = global_params.batch_norm_epsilon;
	DimsHW image_size = DimsHW{global_params.input_h, global_params.input_w};

	std::map<std::string, Weights> weightMap = loadWeights(path_wts);
	Weights emptywts{DataType::kFLOAT, nullptr, 0};
	INetworkDefinition *network = builder->createNetworkV2(0U);
	ITensor *data = network->addInput(INPUT_NAME, dt, Dims3{1, global_params.input_h, global_params.input_w});	// in_channels
	assert(data);

	int out_channels = roundFilters(32, global_params);
        std::cout << "out_channels:" << out_channels <<'\n';	
	auto conv_stem = addSamePaddingConv2d(network, weightMap, *data, out_channels, 3, 2, 1, 1, image_size, "_conv_stem");
	auto bn0 = addBatchNorm2d(network, weightMap, *conv_stem->getOutput(0), "_bn0", bn_eps);
	auto swish0 = addSwish(network, *bn0->getOutput(0));
	ITensor *x = swish0->getOutput(0);
	image_size = calculateOutputImageSize(image_size, 2);
	int block_id = 0;
	for (int i = 0; i < block_args_list.size(); i++)
	{
		BlockArgs block_args = block_args_list[i];

		block_args.input_filters = roundFilters(block_args.input_filters, global_params);
		block_args.output_filters = roundFilters(block_args.output_filters, global_params);
		block_args.num_repeat = roundRepeats(block_args.num_repeat, global_params);
		x = MBConvBlock(network, weightMap, *x, "_blocks." + std::to_string(block_id), block_args, global_params, image_size);

		assert(x);
		block_id++;
		image_size = calculateOutputImageSize(image_size, block_args.stride);
		if (block_args.num_repeat > 1)
		{
			block_args.input_filters = block_args.output_filters;
			block_args.stride = 1;
		}
		for (int r = 0; r < block_args.num_repeat - 1; r++)
		{
			x = MBConvBlock(network, weightMap, *x, "_blocks." + std::to_string(block_id), block_args, global_params, image_size);
			block_id++;
		}
	}
	out_channels = roundFilters(1280, global_params);
	auto conv_head = addSamePaddingConv2d(network, weightMap, *x, out_channels, 1, 1, 1, 1, image_size, "_conv_head", false);
	auto bn1 = addBatchNorm2d(network, weightMap, *conv_head->getOutput(0), "_bn1", bn_eps);
	auto swish1 = addSwish(network, *bn1->getOutput(0));
	auto avg_pool = network->addPoolingNd(*swish1->getOutput(0), PoolingType::kAVERAGE, image_size);

	IFullyConnectedLayer *final = network->addFullyConnected(*avg_pool->getOutput(0), global_params.num_classes, weightMap["_fc.weight"], weightMap["_fc.bias"]);
	assert(final);

	final->getOutput(0)->setName(OUTPUT_NAME);
	network->markOutput(*final->getOutput(0));

	// Build engine
	builder->setMaxBatchSize(maxBatchSize);
	config->setMaxWorkspaceSize(1 << 20);
#ifdef USE_FP16
	config->setFlag(BuilderFlag::kFP16);
#endif
	std::cout << "build engine ..." << std::endl;

	ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
	assert(engine != nullptr);

	std::cout << "build finished" << std::endl;
	// Don't need the network any more
	network->destroy();
	// Release host memory
	for (auto &mem : weightMap)
	{
		free((void *)(mem.second.values));
	}

	return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory **modelStream, std::string wtsPath, std::vector<BlockArgs> block_args_list, GlobalParams global_params)
{
	// Create builder
	IBuilder *builder = createInferBuilder(gLogger);
	IBuilderConfig *config = builder->createBuilderConfig();

	// Create model to populate the network, then set the outputs and create an engine
	ICudaEngine *engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT, wtsPath, block_args_list, global_params);
	assert(engine != nullptr);

	// Serialize the engine
	(*modelStream) = engine->serialize();

	// Close everything down
	engine->destroy();
	builder->destroy();
	config->destroy();
}

void doInference(IExecutionContext &context, float *input, float *output, int batchSize, GlobalParams global_params)
{
	const ICudaEngine &engine = context.getEngine();

	// Pointers to input and output device buffers to pass to engine.
	// Engine requires exactly IEngine::getNbBindings() number of buffers.
	assert(engine.getNbBindings() == 2);
	void *buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// Note that indices are guaranteed to be less than IEngine::getNbBindings()
	const int inputIndex = engine.getBindingIndex(INPUT_NAME);
	const int outputIndex = engine.getBindingIndex(OUTPUT_NAME);

	// Create GPU buffers on device
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 1 * global_params.input_h * global_params.input_w * sizeof(float)));  // in_channels
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * global_params.num_classes * sizeof(float)));

	// Create stream
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 1 * global_params.input_h * global_params.input_w * sizeof(float), cudaMemcpyHostToDevice, stream));	// in_channels(batchSize * 1)
	context.enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * global_params.num_classes * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// Release stream and buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}

bool parse_args(int argc, char **argv, std::string &wts, std::string &engine, std::string &backbone)
{
	if (std::string(argv[1]) == "-s" && argc == 5)
	{
		wts = std::string(argv[2]);
		engine = std::string(argv[3]);
		backbone = std::string(argv[4]);
		std::cout << "Serialize model to engine" << std::endl;
		// ./efficientnet -s xx.wts xx.engine b0-signal
	}
	else if (std::string(argv[1]) == "-d" && argc == 4)
	{
		engine = std::string(argv[2]);
		backbone = std::string(argv[3]);
		// std::cout << "Do Inference" << std::endl;
		// ./efficientnet -d xx.engine b0-signal
		// std::cout << "Do Inference" << "(Params:" << engine << '|' << backbone << '|' << img_path << ')' << std::endl;
	}
	else
	{
		return false;
	}
	return true;
}


std::string wtsPath = "";
std::string engine_name = "";
std::string backbone = "";
std::string img_path = "";

IRuntime *runtime;
ICudaEngine *engine;
IExecutionContext *context;

GlobalParams global_params;

void doGetHi(const httplib::Request& req, httplib::Response& res, const httplib::ContentReader& content_reader)
{

    /********************************** Receive **************************************/
    /*********************************************************************************/

    std::string img_path;
    content_reader([&](const char* data, size_t len){
        img_path.append(data, len);
        return true;
    });

    /******************************* Image prepare************************************/
    /*********************************************************************************/

    auto start = std::chrono::system_clock::now();

    float *data = new float[1 * global_params.input_h * global_params.input_w];

    std::vector<std::vector<float>> mat = read_txt_to_mat(img_path);
    cv::Mat mat_cv(mat.size(), mat[0].size(), CV_32F);
    for (size_t i = 0; i < mat.size(); ++i) {
             for (size_t j = 0; j < mat[i].size(); ++j) {
                         mat_cv.at<float>(i, j) = mat[i][j];
             }
    } 

    std::vector<float> image_data = prepareMatrix(mat_cv, global_params);
    // std::cout << "image_data=" << image_data.size() << std::endl;
    for (int k = 0; k < image_data.size(); k++) {
             data[k] = image_data[k];
     }

    /******************************* Image Inference  ********************************/
    /*********************************************************************************/

    // std::cout << "Num Classes: " << global_params.num_classes << "\n";
   //  auto start = std::chrono::system_clock::now();

    float *prob = new float[global_params.num_classes];
    doInference(*context, data, prob, 1, global_params);

    // Inference out
    float max_confidence = -99;
    int max_index = 0;
    int len_of_prob = (malloc_usable_size(prob) / sizeof(*prob));
    // std::cout << "output len=" << (malloc_usable_size(prob) / sizeof(*prob)) << std::endl;
    for (int k = 0; k < global_params.num_classes; k++) {
              if (prob[k] > max_confidence) {
                max_confidence = prob[k];
		        max_index = k;
              }
    }
    auto end = std::chrono::system_clock::now();
   std::cout << "Infer   " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms "<< std::endl;
    // std::cout << max_index << std::endl;

    // detail inference info
    // std::cout << "Max Confidence: " << max_confidence << std::endl;
    // std::cout << "Max Index: " << max_index << std::endl;
    // std::cout << "Probability: ";
    // for (unsigned int i = 0; i < global_params.num_classes; i++) {
    //         std::cout << prob[i] << '(' << i << ")  ";
    // }
    // std::cout << std::endl;

    /********************************** Response *************************************/
    /*********************************************************************************/
    res.set_content(std::to_string(max_index), "text/plain");
}

int main(int argc, char **argv)
{
        if (!parse_args(argc, argv, wtsPath, engine_name, backbone))
        {
              std::cerr << "arguments not right!" << std::endl;
              std::cerr << "./efficientnet -s [.wts] [.engine] [b0-signal b0 ...] // serialize model to engine" << std::endl;
              std::cerr << "./efficientnet -d [.engine] [b0-signal b0 ...] // startup engine and run inference" << std::endl;
              return -1;
        }
	
	global_params = global_params_map[backbone];
	// create a model using the API directly and serialize it to a stream
	if (!wtsPath.empty())
	{
		IHostMemory *modelStream{nullptr};
		APIToModel(MAX_BATCH_SIZE, &modelStream, wtsPath, block_args_list, global_params);
		assert(modelStream != nullptr);

		std::ofstream p(engine_name, std::ios::binary);
		if (!p)
		{
			std::cerr << "could not open plan output file" << std::endl;
			return -1;
		}
		p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
		modelStream->destroy();
	return 1;
	}

	char *trtModelStream{nullptr};
	size_t size{0};

	std::ifstream file(engine_name, std::ios::binary);
	if (file.good())
	{
		file.seekg(0, file.end);
		size = file.tellg();
		file.seekg(0, file.beg);
		trtModelStream = new char[size];
		assert(trtModelStream);
		file.read(trtModelStream, size);
		file.close();
	}
	else
	{
		std::cerr << "could not open Engine file" << std::endl;
		return -1;
	}

	if (wtsPath.empty()){
		/******************************* Engine start up  ********************************/
		/*********************************************************************************/		
		
		runtime = createInferRuntime(gLogger);
		assert(runtime != nullptr);
		engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
		assert(engine != nullptr);
		context = engine->createExecutionContext();
		assert(context != nullptr);
		delete[] trtModelStream;
		std::cout << "Engine start up " << std::endl;

		
	        // Server
		httplib::Server server;
        	server.Post("/hi", doGetHi);
	        server.listen("0.0.0.0", 8081);


		// Destroy the engine
		context->destroy();
        	engine->destroy();
        	runtime->destroy();
	}
	return 0;
}
