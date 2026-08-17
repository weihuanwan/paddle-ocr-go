package utils

import (
	"fmt"
	"image"

	"github.com/gen2brain/go-fitz"
	"gocv.io/x/gocv"
)

// PDFPageToMat 将 PDF 指定页转换成 gocv.Mat
//
// page 从 0 开始
func PDFPageToMat(pdfPath string, page int) (gocv.Mat, error) {
	doc, err := fitz.New(pdfPath)
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("打开 PDF 失败: %w", err)
	}
	defer doc.Close()

	if page < 0 || page >= doc.NumPage() {
		return gocv.NewMat(), fmt.Errorf(
			"PDF 页码错误: page=%d, 总页数=%d",
			page,
			doc.NumPage(),
		)
	}

	// PDF → image.Image
	img, err := doc.Image(page)
	if err != nil {
		return gocv.NewMat(), fmt.Errorf(
			"PDF 第 %d 页渲染失败: %w",
			page+1,
			err,
		)
	}

	// image.Image → RGBA Mat
	rgbaMat, err := imageToRGBAImage(img)
	if err != nil {
		return gocv.NewMat(), fmt.Errorf(
			"PDF 第 %d 页转换 RGBA Mat 失败: %w",
			page+1,
			err,
		)
	}

	defer rgbaMat.Close()

	// RGBA → BGR
	bgrMat := gocv.NewMat()

	gocv.CvtColor(
		rgbaMat,
		&bgrMat,
		gocv.ColorRGBAToBGR,
	)

	return bgrMat, nil
}

// PDFToMats 将 PDF 所有页面转换成 gocv.Mat
func PDFToMats(pdfPath string) ([]gocv.Mat, error) {
	doc, err := fitz.New(pdfPath)
	if err != nil {
		return nil, fmt.Errorf("打开 PDF 失败: %w", err)
	}
	defer doc.Close()

	mats := make([]gocv.Mat, 0, doc.NumPage())

	for page := 0; page < doc.NumPage(); page++ {

		img, err := doc.Image(page)
		if err != nil {
			// 清理之前的 Mat
			for _, mat := range mats {
				mat.Close()
			}

			return nil, fmt.Errorf(
				"PDF 第 %d 页渲染失败: %w",
				page+1,
				err,
			)
		}

		rgbaMat, err := imageToRGBAImage(img)
		if err != nil {
			for _, mat := range mats {
				mat.Close()
			}

			return nil, fmt.Errorf(
				"PDF 第 %d 页转换失败: %w",
				page+1,
				err,
			)
		}

		bgrMat := gocv.NewMat()

		gocv.CvtColor(
			rgbaMat,
			&bgrMat,
			gocv.ColorRGBAToBGR,
		)

		rgbaMat.Close()

		mats = append(mats, bgrMat)
	}

	return mats, nil
}

// imageToRGBAImage 将 image.Image 转换成 RGBA Mat
func imageToRGBAImage(img image.Image) (gocv.Mat, error) {

	bounds := img.Bounds()

	width := bounds.Dx()
	height := bounds.Dy()

	// go-fitz 返回的 image 通常是 RGBA
	// 但是这里不要直接假设 img.Pix 一定存在。
	// 重新构造 RGBA，兼容 image.Image。
	rgba := image.NewRGBA(image.Rect(0, 0, width, height))

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			rgba.Set(x, y, img.At(
				bounds.Min.X+x,
				bounds.Min.Y+y,
			))
		}
	}

	mat, err := gocv.NewMatFromBytes(
		height,
		width,
		gocv.MatTypeCV8UC4,
		rgba.Pix,
	)

	if err != nil {
		return gocv.NewMat(), err
	}

	return mat, nil
}
