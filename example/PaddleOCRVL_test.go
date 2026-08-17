package example

import (
	"fmt"
	"testing"

	"github.com/weihuanwan/paddleocr-go/layout"
	"github.com/weihuanwan/paddleocr-go/ocr"
	"github.com/weihuanwan/paddleocr-go/vl"
	ort "github.com/yalue/onnxruntime_go"
)

// initSession 抽离公共初始化逻辑
func initSession(t *testing.T) (*vl.PaddleOCRVL, func()) {
	err := ocr.InitOrt("lib/onnxruntime.dll")
	if err != nil {
		t.Fatalf("Error initializing Ort: %v", err)
	}

	options, err := ort.NewSessionOptions()
	if err != nil {
		t.Fatalf("Error creating session options: %v", err)
	}

	layoutDetSessionInternal, err := ort.NewDynamicAdvancedSession(
		"model/PP-DocLayoutV3.onnx",
		[]string{"im_shape", "image", "scale_factor"},
		[]string{"fetch_name_0", "fetch_name_1", "fetch_name_2"},
		options,
	)
	if err != nil {
		options.Destroy()
		t.Fatalf("Error creating layout session: %v", err)
	}

	docLayoutSession := layout.NewLayoutDetSession(layoutDetSessionInternal)

	paddleOCRVL := vl.NewDefaultPaddleOCRVL(
		"PaddlePaddle/PaddleOCR-VL-1.6",
		"http://localhost:8000/v1/chat/completions",
		"sk-ufajxhcyibsxcatybmjqhaierwwbbxjdrhwitcmrscyodhsq",
		docLayoutSession,
	)

	cleanup := func() {
		options.Destroy()
	}
	return paddleOCRVL, cleanup
}

func TestPaddleOCRVL(t *testing.T) {
	session, cleanup := initSession(t)
	defer cleanup()

	imagePath := "pdf/test01.pdf"
	// PDF 会返回多页，内部并发 OCR
	pages, err := session.RunOCR(imagePath)
	if err != nil {
		t.Fatalf("RunOCR failed: %v", err)
	}

	defer session.Close(pages) // 批量释放所有 Mat

	for _, page := range pages {
		t.Logf("===== PDF Page %d | Blocks: %d =====", page.PageIndex+1, len(page.Blocks))
		for _, block := range page.Blocks {
			fmt.Printf("Label: %s | Text: %s\n", block.Label, block.Text)
		}
		fmt.Println()
	}
}
