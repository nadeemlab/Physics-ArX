#!/bin/bash

dataDir=./
inputDir="${dataDir}/Patients/"
outputDir="${dataDir}/Patients/"

patientNames=()
planCTImages=()
CBCTImages=()

patientNames=()
#Read a folder containing patient data
((j=0))
for d in ${inputDir}/*; do
  patientNames[$j]=$(basename $d)
  ((j++))
done

for (( i = 0; i < ${#patientNames[@]}; i++ )) do
  input=${inputDir}/${patientNames[$i]}/
  planCTImages[$i]=${input}/${patientNames[$i]}_CBCTtoCT_plan_cropped_w1.nrrd 
  CBCTImages[$i]=${input}/${patientNames[$i]}_CBCTtoCT_w1_dfWarped.nii.gz
 # planCTLabels_Eso[$i]=${input}/${patientNames[$i]}_CBCTtoCT_plan_Eso_cropped_w1-label.nrrd
done

echo ${#patientNames[@]}
echo ${patientNames[0]}
echo ${patientNames[1]}

echo "Adding scatter artifact from w1CBCT to pCT, then reconstruct using CBCT OS-SART in Matlab ..."
echo

for (( i = 0; i < ${#patientNames[@]}; i++ ))
do
mkdir ${outputDir}/${patientNames[$i]}/Artifacts/
output=${outputDir}/${patientNames[$i]}/Artifacts/
outputPrefix=${output}/${patientNames[$i]}_
mkdir ${outputDir}/${patientNames[$i]}/Recons/
outputPrefixRecon=${outputDir}/${patientNames[$i]}/Recons/

echo ${patientNames[i]} " being processed..."

echo "1)Preprocessing and resampling wk CBCT to pCT..."
echo
ResampleCBCT="./ResampleImages ${planCTImages[$i]} ${CBCTImages[$i]} ${outputPrefix}CBCT_w1.nrrd"
$ResampleCBCT
CBCTImages[$i]="${outputPrefix}CBCT_w1.nrrd"

echo "2)Extract scatters using power-low Adaptive Histogram Equalization matching from CBCT using 7 variations of alpha and beta..."
echo
HistogramEqualization1="./AdaptiveHistogramMatch ${outputPrefix}CBCT_w1.nrrd ${outputPrefix}CBCT_Histogram_1_1.nrrd 5 1 1"
$HistogramEqualization1
echo "alpha=1, beta=1"
echo
HistogramEqualization1="./AdaptiveHistogramMatch ${outputPrefix}CBCT_w1.nrrd ${outputPrefix}CBCT_Histogram_0_1.nrrd 5 0 1"
$HistogramEqualization1
echo "alpha=0, beta=1"
echo

HistogramEqualization2="./AdaptiveHistogramMatch ${outputPrefix}CBCT_w1.nrrd ${outputPrefix}CBCT_Histogram_1_0.nrrd 5 1 0"
$HistogramEqualization2
echo "alpha=1, beta=0"
echo

HistogramEqualization3="./AdaptiveHistogramMatch ${outputPrefix}CBCT_w1.nrrd ${outputPrefix}CBCT_Histogram_0.5_0.5.nrrd 5 0.5 0.5"
$HistogramEqualization3
echo "alpha=0.5, beta=0.5"
echo

HistogramEqualization4="./AdaptiveHistogramMatch ${outputPrefix}CBCT_w1.nrrd ${outputPrefix}CBCT_Histogram_0.5_0.nrrd 5 0.5 0"
$HistogramEqualization4
echo "alpha=0.5, beta=0"
echo

HistogramEqualization5="./AdaptiveHistogramMatch ${outputPrefix}CBCT_w1.nrrd ${outputPrefix}CBCT_Histogram_0_0.5.nrrd 5 0 0.5"
$HistogramEqualization5
echo "alpha=0, beta=0.5"
echo

HistogramEqualization6="./AdaptiveHistogramMatch ${outputPrefix}CBCT_w1.nrrd ${outputPrefix}CBCT_Histogram_0.5_1.nrrd 5 0.5 1"
$HistogramEqualization6
echo "alpha=0.5, beta=1"
echo

HistogramEqualization7="./AdaptiveHistogramMatch ${outputPrefix}CBCT_w1.nrrd ${outputPrefix}CBCT_Histogram_1_0.5.nrrd 5 1 0.5"
$HistogramEqualization7
echo "alpha=1, beta=0.5"
echo

echo "3)Adding extracted artifact to pCT..."
echo
AddImages1="./AddImages ${planCTImages[$i]} ${outputPrefix}CBCT_Histogram_0_1.nrrd ${outputPrefix}pCT_Artifact_0_1.nrrd"
$AddImages1

AddImages2="./AddImages ${planCTImages[$i]} ${outputPrefix}CBCT_Histogram_1_0.nrrd ${outputPrefix}pCT_Artifact_1_0.nrrd"
$AddImages2

AddImages3="./AddImages ${planCTImages[$i]} ${outputPrefix}CBCT_Histogram_0.5_0.5.nrrd ${outputPrefix}pCT_Artifact_0.5_0.5.nrrd"
$AddImages3

AddImages4="./AddImages ${planCTImages[$i]} ${outputPrefix}CBCT_Histogram_0.5_0.nrrd ${outputPrefix}pCT_Artifact_0.5_0.nrrd"
$AddImages4

AddImages5="./AddImages ${planCTImages[$i]} ${outputPrefix}CBCT_Histogram_0_0.5.nrrd ${outputPrefix}pCT_Artifact_0_0.5.nrrd"
$AddImages5

AddImages6="./AddImages ${planCTImages[$i]} ${outputPrefix}CBCT_Histogram_0.5_1.nrrd ${outputPrefix}pCT_Artifact_0.5_1.nrrd"
$AddImages6

AddImages7="./AddImages ${planCTImages[$i]} ${outputPrefix}CBCT_Histogram_1_0.5.nrrd ${outputPrefix}pCT_Artifact_1_0.5.nrrd"
$AddImages7

echo "4)Rescale the images, ready to be sent to Matlab for reconstruction..."
echo
RescalepCT="./RescaleImage ${planCTImages[$i]} ${outputPrefix}rsc_pCT.nrrd 0 1 1"
$RescalepCT
RescaleCBCTImages="./RescaleImage ${CBCTImages[$i]} ${outputPrefix}CBCT_rsc_w1.nrrd 0 1 1"
$RescaleCBCTImages
ResamplepCT="./ResampleImagesFloat ${outputPrefix}CBCT_rsc_w1.nrrd ${outputPrefix}rsc_pCT.nrrd ${outputPrefix}rsc_pCT.nrrd"
$ResamplepCT

#ResampleplabelCBCT="./ResampleLabels ${outputPrefix}CBCT_rsc_w1.nrrd ${planCTLabels_Eso[$i]} ${outputPrefix}CT_plan_Eso_50-label.nrrd 1"
#$ResampleplabelCBCT


RescaleImages1="./RescaleImage ${outputPrefix}pCT_Artifact_0_1.nrrd ${outputPrefix}pCT_Artifact_0_1_rsc.nrrd 0 1 1"
$RescaleImages1

RescaleImages2="./RescaleImage ${outputPrefix}pCT_Artifact_1_0.nrrd ${outputPrefix}pCT_Artifact_1_0_rsc.nrrd 0 1 1"
$RescaleImages2

RescaleImages3="./RescaleImage ${outputPrefix}pCT_Artifact_0.5_0.5.nrrd ${outputPrefix}pCT_Artifact_0.5_0.5_rsc.nrrd 0 1 1"
$RescaleImages3

RescaleImages4="./RescaleImage ${outputPrefix}pCT_Artifact_0.5_0.nrrd ${outputPrefix}pCT_Artifact_0.5_0_rsc.nrrd 0 1 1"
$RescaleImages4

RescaleImages5="./RescaleImage ${outputPrefix}pCT_Artifact_0_0.5.nrrd ${outputPrefix}pCT_Artifact_0_0.5_rsc.nrrd 0 1 1"
$RescaleImages5

RescaleImages6="./RescaleImage ${outputPrefix}pCT_Artifact_0.5_1.nrrd ${outputPrefix}pCT_Artifact_0.5_1_rsc.nrrd 0 1 1"
$RescaleImages6

RescaleImages7="./RescaleImage ${outputPrefix}pCT_Artifact_1_0.5.nrrd ${outputPrefix}pCT_Artifact_1_0.5_rsc.nrrd 0 1 1"
$RescaleImages7

###Reconstruct psCBCT images with OS-SART with separate python script#########
echo "5)Reconstruct OS-SART using separate python script provided in Github repository..."

done
