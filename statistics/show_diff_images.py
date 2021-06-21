# Show slices of difference images with appropriate colormap
import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
plt.rcParams.update({'font.size': 13})

diff_images_dir = './diff_images'
case = '5787'
slice_num = 50

# case = '35414889'
# slice_num = 18

filename = os.path.join(diff_images_dir, case + '_CT.nrrd')
ct = sitk.ReadImage(filename)
ct = sitk.GetArrayFromImage(ct)

filename = os.path.join(diff_images_dir, case + '_CBCT.nrrd')
cbct = sitk.ReadImage(filename)
cbct = sitk.GetArrayFromImage(cbct)

filename = os.path.join(diff_images_dir, case + '_CBCT2CT.nrrd')
sct = sitk.ReadImage(filename)
sct = sitk.GetArrayFromImage(sct)

filename = os.path.join(diff_images_dir, case + '_diff_cbct.nrrd')
cbctdiff = sitk.ReadImage(filename)
cbctdiff = sitk.GetArrayFromImage(cbctdiff)
cbctdiff = np.clip(cbctdiff, a_min=-100, a_max=1200)

filename = os.path.join(diff_images_dir, case + '_diff_cbct2ct.nrrd')
cbct2ctdiff = sitk.ReadImage(filename)
cbct2ctdiff = sitk.GetArrayFromImage(cbct2ctdiff)
cbct2ctdiff = np.clip(cbct2ctdiff, a_min=-400, a_max=400)

# fig = plt.imshow(cbctdiff[slice_num], cmap='jet')
# # fig = plt.imshow(cbct2ctdiff[slice_num], cmap='jet')
# plt.colorbar(fig, fraction=0.046, pad=0.04)
# plt.axis('off')
# plt.title('w1CBCT Difference Image')
# # plt.title('sCT Difference Image')
# # plt.savefig(case + '-cbct2ct-diff.png', bbox_inches='tight', dpi=150)
# plt.savefig(case + '-w1cbct-diff.png', bbox_inches='tight', dpi=150)
# plt.show()

cbctdiff_hist, edges1 = np.histogram(cbctdiff)
cbct2ctdiff_hist, edges2 = np.histogram(cbct2ctdiff)

fig2, ax = plt.subplots(1,1)
ax.plot(edges1[0:-1], cbctdiff_hist, color='r', label='w1CBCT')
ax.plot(edges2[0:-1], cbct2ctdiff_hist, color='b', label='sCT')
ax.legend()
ax.set_title('Difference with pCT image histograms')
ax.set_xlabel('HU value difference', fontsize=16)
ax.set_ylabel('Frequencies', fontsize=16)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

plt.savefig(case + '-diff-histograms.png', bbox_inches='tight', dpi=150)
plt.show()


# fig, axes = plt.subplots(2,3)
# [axi.set_axis_off() for axi in axes.ravel()]
#
# axes[0][0].imshow(ct[slice_num])
#
# axes[0][1].imshow(cbct[slice_num])
# axes[0][2].imshow(sct[slice_num])
#
# img1 = axes[1][1].imshow(cbctdiff[slice_num], cmap='jet')
# fig.colorbar(img1,fraction=0.046, pad=0.04, ax=axes[1][1])
#
# img2 = axes[1][2].imshow(cbct2ctdiff[slice_num], cmap='jet')
# plt.colorbar(img2, shrink=0.55,ax=axes[1][2])
# plt.show()
#


#plt.show()