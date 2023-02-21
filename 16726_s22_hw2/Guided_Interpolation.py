import cv2
import numpy as np
from scipy.signal import convolve2d

test_index = '1'

if test_index == '1':
	roi_tar_x = 110
	roi_tar_y = 50
	IMG_EXTENSIONS = "jpg"
else:
	roi_tar_x = 170
	roi_tar_y = 155
	IMG_EXTENSIONS = "png"

TargetPath = "test" + test_index + "_target."+IMG_EXTENSIONS
SourcePath = "test" + test_index + "_src."+IMG_EXTENSIONS
MaskPath = "test" + test_index + "_mask."+IMG_EXTENSIONS
SaveFilePath = test_index + "."+IMG_EXTENSIONS

kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])

hard = False

# Helper enum
OMEGA = 0
DEL_OMEGA = 1
OUTSIDE = 2

def point_location(index, mask):
	if mask[index].all() == mask[0,0].all():
		return OUTSIDE
	if edge(index,mask) is True:
		return DEL_OMEGA
	return OMEGA

def in_mask(index, mask):
	if mask[index].all() == mask[0,0].all():
		return False
	return True

def edge(index, mask):
	if mask[index].all() == mask[0,0].all():
		return False
	for pt in get_surrounding(index):
		if mask[pt].all() == mask[0,0].all():
			return True
	return False

# Find the indicies of omega, or where the mask is 1
def mask_indicies(mask):
	nonzero = np.nonzero(mask)
	return zip(nonzero[0], nonzero[1])

# Get indicies above, below, to the left and right
def get_surrounding(index):
	i, j = index
	return [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]


if __name__ == '__main__':
	src_img = cv2.imread(SourcePath)
	tar_img = cv2.imread(TargetPath)
	msk_img = cv2.imread(MaskPath)

	# 找到包含mask的最小矩形
	msk_rows, msk_cols = msk_img.shape[:2]
	msk_x, msk_y, msk_indicies = [], [], []
	for i in range(msk_rows):
		for j in range(msk_cols):
			if not msk_img[i][j].all() == msk_img[0][0].all():
				msk_x.append(i)
				msk_y.append(j)
				msk_indicies.append((i,j))
	msk_x_min, msk_x_max, msk_y_min, msk_y_max = min(msk_x), max(msk_x), min(msk_y), max(msk_y)
	roi_rows = msk_x_max-msk_x_min+1
	roi_cols = msk_y_max-msk_y_min+1

	if hard:
		# 直接拼接
		roi_tar = tar_img[roi_tar_x:roi_tar_x+roi_rows, roi_tar_y:roi_tar_y+roi_cols]
		roi_msk_not = cv2.bitwise_not(msk_img[msk_x_min:msk_x_max+1, msk_y_min:msk_y_max+1])
		roi_msk_tar = cv2.bitwise_and(roi_tar, roi_msk_not)
		roi_msk_src = cv2.bitwise_and(msk_img, src_img)[msk_x_min:msk_x_max+1, msk_y_min:msk_y_max+1]
		roi_dst = cv2.add(roi_msk_src, roi_msk_tar)  # 进行融合
		tar_img[roi_tar_x:roi_tar_x+roi_rows, roi_tar_y:roi_tar_y+roi_cols] = roi_dst  # 融合后放在原图上
		cv2.imshow('tar_img',tar_img)
		cv2.imwrite("hard"+SaveFilePath, tar_img)
	else:
		# 对三个通道分别计算
		for color in range(3):
			div_roi_src = convolve2d(src_img[:,:,color], kernel, "same")[msk_x_min:msk_x_max+1, msk_y_min:msk_y_max+1]
			N = len(msk_indicies)  # N = number of points in mask
			A = np.zeros((N, N))
			b = np.zeros(N)
			# Create poisson A and b matrix.
			# Set up row for each point in mask
			for i,index in enumerate(msk_indicies):
				# 边界上的点取target的值
				if point_location(index, msk_img) == DEL_OMEGA:
					A[i,i] = 1
					dst_x = roi_tar_x + index[0] - msk_x_min
					dst_y = roi_tar_y + index[1] - msk_y_min
					b[i] = tar_img[dst_x, dst_y][color]
					continue
				# 非边界的点按照散度列方程
				A[i,i] = -4
				b[i] = div_roi_src[index[0]-msk_x_min, index[1]-msk_y_min]
				for x in get_surrounding(index):
					j = msk_indicies.index(x)
					A[i,j] = 1
			x = np.linalg.solve(A, b)
			for i,index in enumerate(msk_indicies):
				dst_x = roi_tar_x + index[0] - msk_x_min
				dst_y = roi_tar_y + index[1] - msk_y_min
				if x[i] < 0:
					tar_img[dst_x, dst_y][color] = 0
				elif x[i] > 255:
					tar_img[dst_x, dst_y][color] = 255
				else:
					tar_img[dst_x, dst_y][color] = x[i]

		cv2.imwrite(SaveFilePath, tar_img)
		cv2.imshow('tar_img',tar_img)

		cv2.waitKey(0)
		cv2.destroyAllWindows()