#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

struct MNIST {
	MNIST() {
		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
		input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
		output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());
	}

	std::ptrdiff_t Run() {
		const char* input_names[] = { "Input3" };
		const char* output_names[] = { "Plus214_Output_0" };

		session_.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);

		result_ = std::distance(results_.begin(), std::max_element(results_.begin(), results_.end()));
		return result_;
	}

	static constexpr const int width_ = 28;
	static constexpr const int height_ = 28;

	std::array<float, width_* height_> input_image_{};
	std::array<float, 10> results_{};
	int64_t result_{ 0 };

private:
	Ort::Env env;
	Ort::Session session_{ env, L"models/mnist.onnx", Ort::SessionOptions{nullptr}};

	Ort::Value input_tensor_{ nullptr };
	std::array<int64_t, 4> input_shape_{ 1, 1, width_, height_ };

	Ort::Value output_tensor_{ nullptr };
	std::array<int64_t, 2> output_shape_{ 1, 10 };
};


// 画像の読み込み/前処理
std::array<float, 28 * 28> preprocess_image(const std::string& image_path) {
	// 画像をグレースケールで読み込む
	cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
	if (img.empty()) {
		throw std::runtime_error("画像を読み込めませんでした: " + image_path);
	}
	// resize image
	cv::Mat resized_img;
	cv::resize(img, resized_img, cv::Size(28, 28));
	// normalize image
	cv::Mat img_normalized;
	resized_img.convertTo(img_normalized, CV_32F, 1.0 / 255.0);
	// show image
	cv::imshow("normalized", img_normalized);
	cv::waitKey(1000);
	// convert vector
	std::vector<float> input_tensor_values;
	input_tensor_values.assign((float*)img_normalized.datastart, (float*)img_normalized.dataend);
	// convert array
	std::array<float, 28* 28> input_tensor_array;
	std::copy(input_tensor_values.begin(), input_tensor_values.end(), input_tensor_array.begin());
	return input_tensor_array;
}

int main()
{
	std::unique_ptr<MNIST> mnist_;
	try {
		mnist_ = std::make_unique<MNIST>();
	}
	catch (const Ort::Exception& exception) {
		std::cerr << exception.what() << std::endl;
		return 1;
	}
	mnist_->input_image_ = preprocess_image("img/test.png");
	mnist_->Run();
	std::cout << "MNIST Result: " << mnist_->result_ << std::endl;
	return 0;
}