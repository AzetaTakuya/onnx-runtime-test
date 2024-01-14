#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

struct MiDaS
{
	MiDaS() {
		
	}

	void Run() {
		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
		size_t input_tensor_size = 1 * 3 * width_ * height_;
		input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_shape_.data(), input_shape_.size());
		size_t output_tensor_size = 1 * width_ * height_;
		output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());

		const char* input_names[] = { "0" };
		const char* output_names[] = { "1080" };

		session_.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
	}

	static constexpr const int width_ = 384;
	static constexpr const int height_ = 384;

	std::vector<float> input_tensor_values{};
	std::array<float, width_ * height_> results_{};

private:
	Ort::Env env;
	Ort::Session session_{ env, L"models/MiDaS.onnx", Ort::SessionOptions{nullptr} };

	Ort::Value input_tensor_{ nullptr };
	std::array<int64_t, 4> input_shape_{ 1, 3, width_, height_ };

	Ort::Value output_tensor_{ nullptr };
	std::array<int64_t, 3> output_shape_{ 1, width_, height_ };
};

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
std::vector<float> preprocess_image(const std::string& image_path) {
	// 画像をグレースケールで読み込む
	cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
	if (img.empty()) {
		throw std::runtime_error("画像を読み込めませんでした: " + image_path);
	}
	// resize image
	cv::Mat resized_img;
	cv::resize(img, resized_img, cv::Size(384, 384));
	std::cout << "配列サイズ: " << resized_img.size() << std::endl;

	// show image
	cv::imshow("input", resized_img);
	cv::waitKey(1000);
	// convert vector
	std::vector<float> input_tensor_values;
	input_tensor_values.assign(resized_img.datastart, resized_img.dataend);
	std::cout << "配列サイズ: " << input_tensor_values.size() << std::endl;
	return input_tensor_values;
}

int main()
{
	std::unique_ptr<MiDaS> midas_;
	try {
		midas_ = std::make_unique<MiDaS>();
	}
	catch (const Ort::Exception& exception) {
		std::cerr << exception.what() << std::endl;
		return 1;
	}
	midas_->input_tensor_values = preprocess_image("img/test.jpg");
	midas_->Run();
	std::cout << "complete" << std::endl;
	cv::Mat result = cv::Mat(midas_->width_ * midas_->height_, 1, CV_32F, midas_->results_.data());
	double min, max;
	cv::minMaxLoc(result, &min, &max);
	const double range = max - min;
	result.convertTo(result, CV_32F, 1.0 / range, -(min / range));
	result.convertTo(result, CV_8U, 255.0);
	result = result.reshape(0, midas_->width_);
	cv::applyColorMap(result, result, cv::COLORMAP_JET);
	cv::imshow("result",result);
	cv::waitKey(0);
	return 0;
}