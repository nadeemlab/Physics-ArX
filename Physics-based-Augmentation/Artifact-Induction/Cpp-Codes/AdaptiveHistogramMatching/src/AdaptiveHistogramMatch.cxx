#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
//#include "itkAdaptiveEqualizationHistogram.h"
#include "itkAdaptiveHistogramEqualizationImageFilter.h"
//#include "itkFilterWatcher.h"

int
main(int argc, char * argv[])
{
	if (argc < 6)
	{
		std::cerr << "Usage: " << std::endl;
		std::cerr << argv[0] << "  inputImageFile  outputImageFile radius alpha beta" << std::endl;
		return EXIT_FAILURE;
	}


	typedef float InputPixelType;
	static const int ImageDimension = 3;

	typedef itk::Image< InputPixelType, ImageDimension >   InputImageType;
	typedef itk::ImageFileReader< InputImageType >          ReaderType;
	ReaderType::Pointer reader = ReaderType::New();
	reader->SetFileName(argv[1]);
	reader->Update();

	/*typedef itk::AdaptiveHistogramEqualizationImageFilter<
	InputImageType >                           FilterType;*/
	using AdaptiveHistogramEqualizationImageFilterType = itk::AdaptiveHistogramEqualizationImageFilter<InputImageType>;
	AdaptiveHistogramEqualizationImageFilterType::Pointer FilterType =
		AdaptiveHistogramEqualizationImageFilterType::New();
	

	AdaptiveHistogramEqualizationImageFilterType::ImageSizeType radius;
	radius.Fill(atoi(argv[3]));

	FilterType->SetInput(reader->GetOutput());
	FilterType->SetRadius(radius);
	FilterType->SetAlpha(atof(argv[4]));
	FilterType->SetBeta(atof(argv[5]));
	FilterType->Update();
	//
	typedef float WritePixelType;

	typedef itk::Image< WritePixelType, 3 > WriteImageType;

	typedef itk::ImageFileWriter< WriteImageType >  WriterType;

	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(argv[2]);

	try
	{
		writer->SetInput(FilterType->GetOutput());
		std::cout << "Writing Histogram matched image ..." << std::endl;
		writer->Update();
	}
	catch (itk::ExceptionObject & excp)
	{
		std::cerr << "Exception caught " << excp << std::endl;
		return EXIT_FAILURE;
	}
	
	std::cout << "Done!" << std::endl;
	return EXIT_SUCCESS;

}
