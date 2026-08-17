package vl

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"

	"github.com/weihuanwan/paddleocr-go/common"
	"github.com/weihuanwan/paddleocr-go/layout"
	"github.com/weihuanwan/paddleocr-go/utils"
	"gocv.io/x/gocv"
)

type PaddleOCRVL struct {
	Model  string            // 模型名称
	Url    string            // 请求路径
	ApiKey string            // 请求路径
	Tasks  map[string]string //任务类型

	LayoutDetSession *layout.LayoutDetSession //版面分析模型
}

func NewPaddleOCRVLChatCompletionRequest(modelName string,
	dataURL string,
	task string) ChatCompletionRequest {
	messages := []Messages{
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
	return ChatCompletionRequest{
		Model:       modelName,
		Messages:    messages,
		Temperature: 0.0,
	}
}

func NewDefaultPaddleOCRVL(
	model string,
	url string,
	apiKey string,
	layoutDetSession *layout.LayoutDetSession,

) *PaddleOCRVL {
	tasks := map[string]string{
		"ocr":      "OCR:",
		"table":    "Table Recognition:",
		"formula":  "Formula Recognition:",
		"chart":    "Chart Recognition:",
		"seal":     "Seal Recognition:",
		"spotting": "Spotting:",
	}
	paddleOCRVL := &PaddleOCRVL{
		model,
		url,
		apiKey,
		tasks,
		layoutDetSession,
	}

	return paddleOCRVL
}

type PaddleOCRVLBlock struct {
	*common.LayoutDetResult

	OcrResult string
	Text      string
}

func (session *PaddleOCRVL) RunOCR(imagePath string) ([]*PaddleOCRVLBlock, error) {
	originImage, _, err := common.LoadImage(imagePath)
	if err != nil {
		return nil, err
	}
	defer originImage.Close()
	// 版面分析模型识别
	layoutDetResult, err := session.LayoutDetSession.Run(originImage)
	if err != nil {
		return nil, err
	}
	paddleOCRVLBlocks := session.getLayoutParsingResults(layoutDetResult, originImage)
	return paddleOCRVLBlocks, nil
}
func (session *PaddleOCRVL) RunOCRFromMat(originImage *gocv.Mat) ([]*PaddleOCRVLBlock, error) {
	// 版面分析模型识别
	layoutDetResult, err := session.LayoutDetSession.Run(originImage)
	if err != nil {
		return nil, err
	}
	paddleOCRVLBlocks := session.getLayoutParsingResults(layoutDetResult, originImage)
	return paddleOCRVLBlocks, nil
}

func (session *PaddleOCRVL) getLayoutParsingResults(
	layoutDetResult []*common.LayoutDetResult,
	originImage *gocv.Mat,
) []*PaddleOCRVLBlock {

	filterLayoutDetResult := filterOverlapBoxes(layoutDetResult, "auto")
	final := make([]*PaddleOCRVLBlock, 0, len(filterLayoutDetResult))
	for i := 0; i < len(filterLayoutDetResult); i++ {
		detResult := filterLayoutDetResult[i]

		cropImage, err := common.CropByBoxes(detResult, originImage)

		base64Str, err := MatToBase64(cropImage)
		cropImage.Close()
		req := NewPaddleOCRVLChatCompletionRequest(
			session.Model,
			base64Str,
			session.getTask(detResult.Label),
		)

		resp, err := session.Run(req)
		if err != nil {
			log.Fatalf("Error PaddleOCRVL Run: %v", err)
			continue
		}

		ocrResult := resp.Choices[0].Message.Content
		text := ocrResult

		if detResult.Label == "table" {
			text = utils.ConvertOtslToHtml(ocrResult)
		}

		block := &PaddleOCRVLBlock{
			LayoutDetResult: detResult,
			OcrResult:       ocrResult,
			Text:            text,
		}
		final = append(final, block)

	}

	return final
}

func (session *PaddleOCRVL) getTask(key string) string {
	task := session.Tasks[key]
	if task == "" {
		task = "OCR:"
	}
	return task
}
func MatToBase64(mat *gocv.Mat) (string, error) {

	buf, err := gocv.IMEncode(".png", *mat)
	if err != nil {
		return "", err
	}
	defer buf.Close()
	// 2. 转 base64
	base64Str := "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.GetBytes())
	return base64Str, nil
}
func (session *PaddleOCRVL) Run(request ChatCompletionRequest) (*ChatCompletionResponse, error) {

	request.Model = session.Model
	reqBody, err := json.Marshal(request)
	// 5️⃣ 构造 HTTP 请求（标准写法）
	req, err := http.NewRequest(
		"POST",
		session.Url,
		bytes.NewBuffer(reqBody),
	)
	if err != nil {
		return nil, fmt.Errorf("req error %s  %s", session.Url, err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", session.ApiKey)

	// 6️⃣ 发请求
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("send api error %s  %s", session.Url, err)
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
