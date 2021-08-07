#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
//#include "itkAdaptiveHistogramEqualizationImageFilter.h"
//#include "itkFilterWatcher.h"
#include "itkResampleImageFilter.h"

int
main(int argc, char * argv[])
{
	if (argc < 5)
	{
		std::cerr << "Usage: " << std::endl;
		std::cerr << argv[0] << "  inputImageFile  outputImageFile MinRange MaxRange IsISO " << std::endl;
		return EXIT_FAILURE;
	}
	
	typedef float InputPixelType;
	static const int ImageDimension = 3;

	typedef itk::Image< InputPixelType, ImageDimension >   InputImageType;
	typedef InputImageType::SpacingType    SpacingType;
	typedef InputImageType::PointType      OriginType;
	typedef InputImageType::RegionType     RegionType;
	typedef InputImageType::SizeType       SizeType;
	
	typedef itk::ImageFileReader< InputImageType >          ReaderType;
	                        
	ReaderType::Pointer reader = ReaderType::New();
	reader->SetFileName(argv[1]);
	reader->Update();

	typedef float WritePixelType;
	typedef itk::Image< WritePixelType, 3 > WriteImageType;
		
	InputImageType::Pointer Image = InputImageType::New();
	Image = reader->GetOutput();
	SpacingType inputImageSpacing = Image->GetSpacing();
	
	WriteImageType::Pointer OutImage = WriteImageType::New();

	unsigned char IsIso = atoi(argv[5]);

	if (IsIso == 1)
	{
		SpacingType inputImageSpacing = Image->GetSpacing();
		OriginType  inputImageOrigin = Image->GetOrigin();
		RegionType  inputImageRegion = Image->GetLargestPossibleRegion();
		SizeType    inputImageSize = inputImageRegion.GetSize();
		//InputImageType::DirectionType inputImageDirection = RescaledImage->GetDirection();

		typedef itk::ResampleImageFilter<
			InputImageType, WriteImageType >  ResampleFilterTypeISO;
		ResampleFilterTypeISO::Pointer resampleriso = ResampleFilterTypeISO::New();
		resampleriso->SetInput(Image);
		
		typedef itk::IdentityTransform< double, ImageDimension >  TransformType;
		TransformType::Pointer transform = TransformType::New();
		transform->SetIdentity();
		resampleriso->SetTransform(transform);
		//std::cout << "Input Image spacing = " << double(inputImageSpacing[0]) << std::endl;
		typedef itk::LinearInterpolateImageFunction<
			InputImageType, double >  InterpolatorType;
		InterpolatorType::Pointer interpolator = InterpolatorType::New();
		resampleriso->SetInterpolator(interpolator);
		resampleriso->SetDefaultPixelValue(0); // highlight regions without source
		SpacingType spacing;
		spacing[0] = inputImageSpacing[0];
		spacing[1] = inputImageSpacing[1];
		spacing[2] = inputImageSpacing[0];
		/*spacing[0] = 1.0;
		spacing[1] = 1.0;
		spacing[2] = 1.0;*/
		resampleriso->SetOutputSpacing(spacing);
		resampleriso->SetOutputOrigin(inputImageOrigin);
		resampleriso->SetOutputDirection(Image->GetDirection());
		//std::cout << "Input Image spacing = " << inputImageSpacing[3] << std::endl;
		//InputImageType::SizeType   inputSize = inputImageSize;
		typedef InputImageType::SizeType::SizeValueType SizeValueType;
		const double dx = inputImageSize[0] * inputImageSpacing[0] / inputImageSpacing[0];
		const double dy = inputImageSize[1] * inputImageSpacing[1] / inputImageSpacing[1];
		const double dz = inputImageSize[2] * inputImageSpacing[2] / inputImageSpacing[0];
		/*const double dx = inputImageSize[0] * inputImageSpacing[0];
		const double dy = inputImageSize[1] * inputImageSpacing[1];
		const double dz = inputImageSize[2] * inputImageSpacing[2];*/
		
		SizeType   size;
		size[0] = static_cast<SizeValueType>(dx);
		size[1] = static_cast<SizeValueType>(dy);
		size[2] = static_cast<SizeValueType>(dz);
		resampleriso->SetSize(size);
		resampleriso->Update();

		OutImage = resampleriso->GetOutput();
	}
	//WriteImageFile<InputImageType>(inputImage, "testiso.nrrd");
	
	/////////Isotropic voxels//////////////
	typedef itk::RescaleIntensityImageFilter<
		WriteImageType, WriteImageType > RescaleFilterType;
	RescaleFilterType::Pointer rescaler = RescaleFilterType::New();
	rescaler->SetOutputMinimum(std::stoi(argv[3]));
	rescaler->SetOutputMaximum(std::stoi(argv[4]));
	rescaler->SetInput(OutImage);

	WriteImageType::Pointer RescaledImage = WriteImageType::New();
	RescaledImage = rescaler->GetOutput();
	
	typedef itk::ImageFileWriter< WriteImageType >  WriterType;
	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(argv[2]);
	
	try
	{
		writer->SetInput(RescaledImage);
		std::cout << "Writing rescaled image ..." << std::endl;
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
