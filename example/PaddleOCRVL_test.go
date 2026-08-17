package example

import (
	"fmt"
	"log"
	"os"
	"testing"

	"github.com/weihuanwan/paddleocr-go/layout"
	"github.com/weihuanwan/paddleocr-go/ocr"
	"github.com/weihuanwan/paddleocr-go/utils"
	"github.com/weihuanwan/paddleocr-go/vl"
	ort "github.com/yalue/onnxruntime_go"
)

func TestOcrImage(t *testing.T) {
	wd, _ := os.Getwd()
	t.Logf("===== 当前工作目录: %s =====", wd)
	err := ocr.InitOrt("lib/onnxruntime.dll")
	if err != nil {

		log.Fatalf("Error initializing Ort: %v", err)
		panic(err)
	}

	options, _ := ort.NewSessionOptions()
	defer options.Destroy()
	// CLS
	layoutDetSessionInternal, err := ort.NewDynamicAdvancedSession(
		"model/PP-DocLayoutV3.onnx",
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

	imagePath := "images/test.jpg"
	paddleOCRVLBlocks, err := paddleOCRVL.RunOCR(imagePath)
	if err != nil {
		panic(err)
	}

	for i := 0; i < len(paddleOCRVLBlocks); i++ {
		block := paddleOCRVLBlocks[i]
		fmt.Println(block.Label)
		fmt.Println(block.Text)
	}
}

func TestOcrPdf(t *testing.T) {

	err := ocr.InitOrt("lib/onnxruntime.dll")
	if err != nil {
		log.Fatalf("Error initializing Ort: %v", err)
	}

	options, _ := ort.NewSessionOptions()
	defer options.Destroy()
	// CLS
	layoutDetSessionInternal, err := ort.NewDynamicAdvancedSession(
		"model/PP-DocLayoutV3.onnx",
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

	imagePath := "pdf/test01.pdf"

	mats, err := utils.PDFToMats(imagePath)

	if err != nil {
		panic(err)
	}

	for i := 0; i < len(mats); i++ {
		paddleOCRVLBlocks, err := paddleOCRVL.RunOCRFromMat(&mats[i])

		if err != nil {
			panic(err)
		}

		for i := 0; i < len(paddleOCRVLBlocks); i++ {
			block := paddleOCRVLBlocks[i]
			fmt.Println(block.Label)
			fmt.Println(block.Text)

		}
	}
}
