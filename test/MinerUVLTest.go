package main

import (
	"fmt"

	"github.com/weihuanwan/paddleocr-go/vl"
)

func main() {

	minerUVl := vl.NewDefaultMinerUVL("OpenDataLab/MinerU2.5-Pro-2605-1.2B",
		"http://localhost:8000/v1/chat/completions", "sk-ufajxhcyibsxcatybmjqhaierwwbbxjdrhwitcmrscyodhsq")

	imagePath := "test/images/layout0.png"
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
