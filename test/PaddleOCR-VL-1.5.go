package main

import (
	"fmt"
	"log"

	"github.com/gen2brain/go-fitz"
	"github.com/weihuanwan/paddleocr-go/layout"
	"github.com/weihuanwan/paddleocr-go/ocr"
	"github.com/weihuanwan/paddleocr-go/vl"
	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"
)

func main() {

	err := ocr.InitOrt("./test/lib/onnxruntime.dll")
	if err != nil {
		log.Fatalf("Error initializing Ort: %v", err)
	}

	options, _ := ort.NewSessionOptions()
	defer options.Destroy()
	// CLS
	layoutDetSessionInternal, err := ort.NewDynamicAdvancedSession(
		"test/model/PP-DocLayoutV3.onnx",
		[]string{"im_shape", "image", "scale_factor"},
		[]string{"fetch_name_0", "fetch_name_1", "fetch_name_2"},
		options,
	)
	if err != nil {
		panic(err)
	}

	docLayoutSession := layout.NewLayoutDetSession(layoutDetSessionInternal)

	paddleOCRVL := vl.NewDefaultPaddleOCRVL("PaddlePaddle/PaddleOCR-VL-1.6",
		"http://localhost:8000/v1/chat/completions", "sk-ufajxhcyibsxcatybmjqhaierwwbbxjdrhwitcmrscyodhsq", docLayoutSession)

	doc, err := fitz.New("C:\\Users\\Administrator\\Desktop\\word\\paddleocr-go\\test\\CIntegratedBankStatement_HK_124861001279773756J84CQn_20250630.pdf")

	if err != nil {
		panic(err)
	}
	defer doc.Close()

	// 2. 遍历每一页
	for n := 0; n < doc.NumPage(); n++ {
		img, err := doc.Image(n)
		if err != nil {
			panic(err)
		}

		bounds := img.Bounds()

		originImage, err := gocv.NewMatFromBytes(
			bounds.Dy(), bounds.Dx(),
			gocv.MatTypeCV8UC4,
			img.Pix,
		)
		if err != nil {
			panic(err)

		}

		// RGBA → RGB（3 通道）
		rgbImage := gocv.NewMat()
		gocv.CvtColor(originImage, &rgbImage, gocv.ColorRGBAToBGR) // 或 ColorRGBAToRGB
		defer rgbImage.Close()

		if err != nil {
			panic(err)
		}
		w1 := gocv.NewWindow("originImage")
		w1.ResizeWindow(bounds.Dx(), bounds.Dy())
		w1.IMShow(rgbImage)
		w1.WaitKey(0)
		paddleOCRVLBlocks, err := paddleOCRVL.RunOCRFromMat(&rgbImage)

		if err != nil {
			panic(err)
		}
		for i := 0; i < len(paddleOCRVLBlocks); i++ {
			block := paddleOCRVLBlocks[i]
			fmt.Println(block.Label)
			fmt.Println(block.Text)

		}
	}

	//
	//imageMat := gocv.IMRead(imagePath, gocv.IMReadColor)
	//defer imageMat.Close()

	// 8️⃣ 输出结果
	//fmt.Println(result)
}

//
