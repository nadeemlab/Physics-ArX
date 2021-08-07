#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
///////Masking initialization/////////////////////
const    unsigned int    ImageDimension = 3;
typedef  float PixelType;
typedef  float OutputPixelType;
typedef itk::Image< OutputPixelType, ImageDimension >  FixedImageType;
typedef FixedImageType::SpacingType    SpacingType;
typedef FixedImageType::PointType      OriginType;
typedef FixedImageType::RegionType     RegionType;
typedef FixedImageType::SizeType       SizeType;
typedef FixedImageType::IndexType      IndexType;

int main( int argc, char *argv[] )
{
  if( argc < 3 ) //accept minimum 2 values
    {
    std::cerr << "Missing Parameters " << std::endl;
    std::cerr << "Usage: " << argv[0];
    std::cerr << " ReferenceImage ImagetobeResampled FinalResampledImage";
	std::cerr << std::endl;
    return EXIT_FAILURE;
    }

 // typedef itk::Image< PixelType, ImageDimension >  LabelImageType;
  typedef itk::Image< OutputPixelType, ImageDimension > OutputImageType;
   //
  //   Read the Fixed and Moving images.
  //
  typedef itk::ImageFileReader< FixedImageType  > FixedImageReaderType;
  typedef itk::ImageFileReader< FixedImageType > ResampleImageReaderType;
  
  FixedImageReaderType::Pointer  fixedImageReader  = FixedImageReaderType::New();
  ResampleImageReaderType::Pointer resampleImageReader = ResampleImageReaderType::New();
  
  fixedImageReader->SetFileName(  argv[1] );
  resampleImageReader->SetFileName( argv[2] );
 
  try
    {
    fixedImageReader->Update();
	resampleImageReader->Update();
    }

  catch( itk::ExceptionObject & err )
    {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }

  FixedImageType::Pointer fixedImage = fixedImageReader->GetOutput();
  FixedImageType::Pointer resampleImage = resampleImageReader->GetOutput();
  
  /*typedef itk::ImageRegionIterator< LabelImageType > IteratorTypeLabelImage;
  IteratorTypeLabelImage itlabelimage(labelImage, labelImage->GetLargestPossibleRegion());
  itlabelimage.GoToBegin();

  int color = 1;
  if (argc > 4)
  {
	  color = atoi(argv[4]);
  }

  while (!itlabelimage.IsAtEnd())
  {
	  if (itlabelimage.Get() > 0)
	  {
		  itlabelimage.Set(color);
	  }
	  
	  ++itlabelimage;
  }*/
  
  // FixedImageType::RegionType fixedRegion = fixedImage->GetBufferedRegion();

  ////////Resampling process- Start/////////////
  /*typedef itk::CastImageFilter<
	  MovingImageType,
	  OutputImageType > CastFilterType;
  CastFilterType::Pointer  caster1 = CastFilterType::New();
  caster1->SetInput(movingImage);*/

  typedef itk::ResampleImageFilter<
	  FixedImageType,
	  FixedImageType>    ResampleImageFilterType;

  ResampleImageFilterType::Pointer resamplemovingimage = ResampleImageFilterType::New();
  resamplemovingimage->SetInput(resampleImage);
  resamplemovingimage->SetSize(fixedImage->GetLargestPossibleRegion().GetSize());
  resamplemovingimage->SetOutputOrigin(fixedImage->GetOrigin());
  resamplemovingimage->SetOutputSpacing(fixedImage->GetSpacing());
  resamplemovingimage->SetOutputDirection(fixedImage->GetDirection());
  typedef itk::LinearInterpolateImageFunction<
	  FixedImageType, double >  InterpolatorType;
  InterpolatorType::Pointer interpolator1 = InterpolatorType::New();
  resamplemovingimage->SetInterpolator(interpolator1);
  resamplemovingimage->Update();
  resampleImage = resamplemovingimage->GetOutput();
  
  typedef itk::ImageFileWriter< OutputImageType >  WriterType;
  WriterType::Pointer      writer1 = WriterType::New();
  writer1->SetFileName(argv[3]);
  
  try
  {
	  writer1->SetInput(resampleImage);
	  std::cout << "Writing resampled float image...";
	  writer1->Update();
  }
  catch (itk::ExceptionObject & err)
  {
	  std::cerr << "ExceptionObject caught !" << std::endl;
	  std::cerr << err << std::endl;
	  return EXIT_FAILURE;
  }

  std::cout << " Done!" << std::endl;


 return EXIT_SUCCESS;
}
