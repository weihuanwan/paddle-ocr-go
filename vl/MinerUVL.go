package vl

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"io"
	"net/http"

	"github.com/weihuanwan/paddleocr-go/common"
	"github.com/weihuanwan/paddleocr-go/layout"
	"gocv.io/x/gocv"
)

type MinerUVl struct {
	Model            string            // 模型名称
	Url              string            // 请求路径
	ApiKey           string            // 请求路径
	Tasks            map[string]string //任务类型
	LayoutImageSize  [2]int
	LayoutDetSession *layout.LayoutDetSession //版面分析模型
}

func NewMinerUVlChatCompletionRequest(modelName string,
	dataURL string,
	task string) ChatCompletionRequest {
	messages := []Messages{
		{
			Role:    "system",
			Content: "You are a helpful assistant.",
		},
		{
			Role: "user",
			Content: []Content{
				{
					Type: "image_url",
					ImageURL: &ImageURL{
						URL: dataURL,
					},
				},
				{
					Type: "text",
					Text: task,
				},
			},
		},
	}
	vllmXargs := map[string]interface{}{
		"no_repeat_ngram_size": 100,
		"debug":                false,
	}

	return ChatCompletionRequest{
		Model:             modelName,
		Messages:          messages,
		Temperature:       0.0,
		TopK:              1,
		TopP:              0.01,
		PresencePenalty:   1,
		FrequencyPenalty:  0.05,
		RepetitionPenalty: 1,
		SkipSpecialTokens: false,
		VllmXargs:         vllmXargs,
	}
}

func NewDefaultMinerUVL(
	model string,
	url string,
	apiKey string,
	layoutDetSession *layout.LayoutDetSession,

) *MinerUVl {
	tasks := map[string]string{
		"table":                    "\nTable Recognition:",
		"equation":                 "\nFormula Recognition:",
		"image":                    "\nImage Analysis:",
		"chart":                    "\nImage Analysis:",
		"[default]":                "\nText Recognition:",
		"[layout]":                 "\nLayout Detection:",
		"[cross_page_table_merge]": "",
	}
	MinerUVl := &MinerUVl{
		model,
		url,
		apiKey,
		tasks,
		[2]int{1036, 1036},
		layoutDetSession,
	}

	return MinerUVl
}

type MinerUVlBlock struct {
	*common.LayoutDetResult

	OcrResult string
	Text      string
}

func (session *MinerUVl) RunOCR(imagePath string) ([]*MinerUVlBlock, error) {
	originImage, _, err := common.LoadImage(imagePath)
	if err != nil {
		return nil, err
	}
	defer originImage.Close()

	//layoutDetResult, err := session.layoutDetect(originImage)

	//// 版面分析模型识别
	layoutDetResult, err := session.LayoutDetSession.Run(originImage)
	if err != nil {
		return nil, err
	}
	MinerUVlBlocks := session.getLayoutParsingResults(layoutDetResult, originImage)
	return MinerUVlBlocks, nil
}

func (session *MinerUVl) getLayoutParsingResults(
	layoutDetResult []*common.LayoutDetResult,
	originImage *gocv.Mat,
) []*MinerUVlBlock {

	filterLayoutDetResult := filterOverlapBoxes(layoutDetResult, "auto")
	final := make([]*MinerUVlBlock, 0, len(filterLayoutDetResult))
	for i := 0; i < len(filterLayoutDetResult); i++ {
		detResult := filterLayoutDetResult[i]

		cropImage, err := common.CropByBoxes(detResult, originImage)

		base64Str, err := matToBase64(cropImage)
		cropImage.Close()
		req := NewMinerUVlChatCompletionRequest(
			session.Model,
			base64Str,
			session.getTask(detResult.Label),
		)

		resp, err := session.predict(req)
		if err != nil {
			continue
		}

		ocrResult := resp.Choices[0].Message.Content
		text := ocrResult

		if detResult.Label == "table" {
			text = ConvertOtslToHtml(ocrResult)
		}

		block := &MinerUVlBlock{
			LayoutDetResult: detResult,
			OcrResult:       ocrResult,
			Text:            text,
		}
		final = append(final, block)

	}

	return final
}

func (session *MinerUVl) getTask(key string) string {
	task := session.Tasks[key]
	if task == "" {
		task = "\nText Recognition:"
	}
	return task
}

func (session *MinerUVl) predict(request ChatCompletionRequest) (*ChatCompletionResponse, error) {
	request.Model = session.Model
	reqBody, err := json.Marshal(request)
	// 5️⃣ 构造 HTTP 请求（标准写法）
	req, err := http.NewRequest(
		"POST",
		session.Url,
		bytes.NewBuffer(reqBody),
	)
	if err != nil {

		return nil, fmt.Errorf("send api error %s  %s", session.Url, err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+session.ApiKey)

	// 6️⃣ 发请求
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	// 7️⃣ 读取返回
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read body error %s  %s", session.Url, err)
	}

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("请求失败")
	}
	var result ChatCompletionResponse
	err = json.Unmarshal(body, &result)

	return &result, nil

}

func (session *MinerUVl) layoutDetect(originImage *gocv.Mat) ([]*common.LayoutDetResult, error) {
	resizeMat := gocv.NewMat()
	defer resizeMat.Close()
	err := gocv.Resize(*originImage, &resizeMat, image.Pt(session.LayoutImageSize[0], session.LayoutImageSize[1]), 0, 0, gocv.InterpolationCubic)

	if err != nil {
		return nil, fmt.Errorf("session resize failed: %v", err)
	}

	//scaleW := float32(session.LayoutImageSize[0]) / float32(originImage.Cols())
	//scaleH := float32(session.LayoutImageSize[1]) / float32(originImage.Rows())

	base64Str, err := matToBase64(&resizeMat)

	req := NewMinerUVlChatCompletionRequest(
		session.Model,
		base64Str,
		session.getTask("[layout]"),
	)

	resp, err := session.predict(req)
	if err != nil {
		return nil, fmt.Errorf("session predict [layout]  failed: %v", err)
	}
	ocrResult := resp.Choices[0].Message.Content
	println(ocrResult)

	return nil, nil
}
