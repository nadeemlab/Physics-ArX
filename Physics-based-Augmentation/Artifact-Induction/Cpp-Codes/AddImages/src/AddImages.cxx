#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkAddImageFilter.h"
//#include "itkAdaptiveHistogramEqualizationImageFilter.h"
//#include "itkFilterWatcher.h"
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"

int
main(int argc, char * argv[])
{
	if (argc < 3)
	{
		std::cerr << "Usage: " << std::endl;
		std::cerr << argv[0] << "  inputImageFile1  inputImageFile2 outputImageFile " << std::endl;
		return EXIT_FAILURE;
	}
	
	static const int ImageDimension = 3;
	typedef float InputPixelType;
	typedef itk::Image< InputPixelType, ImageDimension >   InputImageType;
	typedef itk::Image< InputPixelType, 3 > OutputImageType;
	
	typedef itk::ImageFileReader< InputImageType >          InputReaderType;
	typedef itk::ImageFileReader< OutputImageType >          OutputReaderType;

	InputReaderType::Pointer reader1 = InputReaderType::New();
	reader1->SetFileName(argv[1]);
	reader1->Update();

	InputReaderType::Pointer reader2 = InputReaderType::New();
	reader2->SetFileName(argv[2]);
	reader2->Update();

	InputImageType::Pointer Image1 = InputImageType::New();
	InputImageType::Pointer Image2 = InputImageType::New();
	Image1 = reader1->GetOutput();
	Image2 = reader2->GetOutput();

	typedef itk::ResampleImageFilter<
		InputImageType,
		InputImageType>    ResampleImageFilterType;

	ResampleImageFilterType::Pointer resamplemovingimage = ResampleImageFilterType::New();
	resamplemovingimage->SetInput(Image2);
	resamplemovingimage->SetSize(Image1->GetLargestPossibleRegion().GetSize());
	resamplemovingimage->SetOutputOrigin(Image1->GetOrigin());
	resamplemovingimage->SetOutputSpacing(Image1->GetSpacing());
	resamplemovingimage->SetOutputDirection(Image1->GetDirection());
	typedef itk::LinearInterpolateImageFunction<
		InputImageType, double >  InterpolatorType;
	InterpolatorType::Pointer interpolator1 = InterpolatorType::New();
	resamplemovingimage->SetInterpolator(interpolator1);
	resamplemovingimage->Update();
	Image2 = resamplemovingimage->GetOutput();
	
	using AddImageFilterType = itk::AddImageFilter<InputImageType, OutputImageType>;

	AddImageFilterType::Pointer addFilter = AddImageFilterType::New();
	addFilter->SetInput1(Image1);
	addFilter->SetInput2(Image2);
	addFilter->Update();

	typedef itk::ImageFileWriter< OutputImageType >  WriterType;
	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(argv[3]);
	
	try
	{
		writer->SetInput(addFilter->GetOutput());
		std::cout << "Writing Added image ..." << std::endl;
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
