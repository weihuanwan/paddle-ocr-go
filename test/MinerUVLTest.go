package main

import (
	"fmt"
	"log"

	"github.com/weihuanwan/paddleocr-go/layout"
	"github.com/weihuanwan/paddleocr-go/ocr"
	"github.com/weihuanwan/paddleocr-go/vl"
	ort "github.com/yalue/onnxruntime_go"
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

	minerUVl := vl.NewDefaultMinerUVL("OpenDataLab/MinerU2.5-Pro-2605-1.2B",
		"http://localhost:8000/v1/chat/completions", "sk-ufajxhcyibsxcatybmjqhaierwwbbxjdrhwitcmrscyodhsq", docLayoutSession)

	imagePath := "test/images/table_recognition2.jpg"
	//
	//imageMat := gocv.IMRead(imagePath, gocv.IMReadColor)
	//defer imageMat.Close()

	paddleOCRVLBlocks, err := minerUVl.RunOCR(imagePath)

	if err != nil {
		panic(err)
	}
	for i := 0; i < len(paddleOCRVLBlocks); i++ {
		block := paddleOCRVLBlocks[i]
		fmt.Println(block.Label)
		fmt.Println(block.Text)
		fmt.Println("----------------------------")
	}

	// 8️⃣ 输出结果
	//fmt.Println(result)
}

//
