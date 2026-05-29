package vl

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"io"
	"math"
	"net/http"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/weihuanwan/paddleocr-go/common"
	"gocv.io/x/gocv"
)

type MinerUVl struct {
	Model           string            // 模型名称
	Url             string            // 请求路径
	ApiKey          string            // 请求路径
	Tasks           map[string]string //任务类型
	LayoutImageSize [2]int
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

	layoutDetResult, err := session.layoutDetect(originImage)

	minerUVlBlocks := session.getLayoutParsingResults(layoutDetResult, originImage)
	return minerUVlBlocks, nil
}

func (session *MinerUVl) getLayoutParsingResults(
	layoutDetResult []*common.LayoutDetResult,
	originImage *gocv.Mat,
) []*MinerUVlBlock {

	layoutDetResult = filterInternalLayoutBlocks(layoutDetResult,
		[]string{"image", "chart"},
		[]string{"text", "equation", "image_block"},
		0.9,
	)

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

var blockRe = regexp.MustCompile(
	`^<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)` +
		`<\|box_end\|><\|ref_start\|>(\w+)<\|ref_end\|>` +
		`(?:<\|rotate_(up|right|down|left)\|>)?` +
		`([\s\S]*)$`,
)
var rotateAngles = map[string]int{
	"up":    0,
	"right": 90,
	"down":  180,
	"left":  270,
}
var blockTypes = map[string]struct{}{
	"algorithm":      {},
	"aside_text":     {},
	"chart":          {},
	"code":           {},
	"code_caption":   {},
	"equation":       {},
	"equation_block": {},
	"footer":         {},
	"header":         {},
	"image":          {},
	"image_block":    {},
	"image_caption":  {},
	"image_footnote": {},
	"list":           {},
	"list_item":      {},
	"page_footnote":  {},
	"page_number":    {},
	"phonetic":       {},
	"ref_text":       {},
	"table":          {},
	"table_caption":  {},
	"table_footnote": {},
	"text":           {},
	"title":          {},
	"unknown":        {},
}

func (session *MinerUVl) layoutDetect(originImage *gocv.Mat) ([]*common.LayoutDetResult, error) {
	// 1. resize 1036,1036
	resizeMat := gocv.NewMat()
	height := originImage.Rows()
	width := originImage.Cols()
	defer resizeMat.Close()
	err := gocv.Resize(*originImage, &resizeMat, image.Pt(session.LayoutImageSize[0], session.LayoutImageSize[1]), 0, 0, gocv.InterpolationCubic)

	if err != nil {
		return nil, fmt.Errorf("session resize failed: %v", err)
	}

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
	output := resp.Choices[0].Message.Content
	if output == "" {
		return nil, nil
	}

	// 注意：strings.Split 会去掉分隔符，需要加回来
	parts := strings.Split(output, "<|box_start|>")

	layoutDetResults := make([]*common.LayoutDetResult, 0, len(parts))
	for i, part := range parts {
		// 第一部分：如果 output 不以 <|box_start|> 开头，这里是前缀垃圾
		if i == 0 && strings.TrimSpace(part) != "" {
			fmt.Println("Layout output prefix not matching expected format", "content", part)
			continue
		}
		// 恢复完整块
		blockStr := "<|box_start|>" + part
		blockStr = strings.TrimSpace(blockStr)
		m := blockRe.FindStringSubmatch(blockStr)
		if m == nil {
			fmt.Println("Layout output does not match expected format", "block", blockStr)
			continue
		}

		// m[1]~m[4]: x1, y1, x2, y2
		x1, _ := strconv.Atoi(m[1])
		y1, _ := strconv.Atoi(m[2])
		x2, _ := strconv.Atoi(m[3])
		y2, _ := strconv.Atoi(m[4])

		// 确保 x1 < x2, y1 < y2
		if x2 < x1 {
			x1, x2 = x2, x1
		}
		if y2 < y1 {
			y1, y2 = y2, y1
		}
		x1 = int(math.Round(float64(x1) / 1000.0 * float64(width)))
		y1 = int(math.Round(float64(y1) / 1000.0 * float64(height)))
		x2 = int(math.Round(float64(x2) / 1000.0 * float64(width)))
		y2 = int(math.Round(float64(y2) / 1000.0 * float64(height)))

		// m[5]: ref_type
		refType := strings.ToLower(m[5])
		if refType == "unknown" {
			refType = "image"
		}
		if refType == "inline_formula" {
			fmt.Println("Skipping inline formula block in layout output", "block", blockStr)
			continue
		}
		if _, ok := blockTypes[refType]; !ok {
			fmt.Println("Unknown block type in layout output line", "block", blockStr)
			continue
		}
		// m[6]: 旋转方向（如 "up"），可能为空
		var angle *int
		if m[6] != "" {
			if deg, ok := rotateAngles[m[6]]; ok {
				angle = &deg
			}
		}
		if angle == nil {
			fmt.Println("No angle found in layout output line", "block", blockStr)
			continue
		}

		tail := strings.TrimSpace(m[7])
		// text 类型特殊处理 merge_prev
		if refType == "text" {
			fmt.Println(tail)
		}
		layoutDetResult := &common.LayoutDetResult{
			Label: refType,
			Point: []int{x1, y1, x2, y2},
		}

		layoutDetResults = append(layoutDetResults, layoutDetResult)
	}
	layoutDetResults = filterInternalLayoutBlocks(layoutDetResults, []string{"table"},
		[]string{"text", "equation", "equation_block"},
		0.9)
	return layoutDetResults, nil
}
func filterInternalLayoutBlocks(blocks []*common.LayoutDetResult, containerTypes []string,
	candidateTypes []string,

	threshold float64) []*common.LayoutDetResult {

	blockIndex := common.FindCoveredBlockIndices(blocks,
		containerTypes,
		candidateTypes,
		threshold,
	)
	if len(blockIndex) == 0 {
		return blocks
	}
	layoutDetResults := make([]*common.LayoutDetResult, 0, len(blocks)-len(blockIndex))
	for i := 0; i < len(blocks); i++ {
		if !slices.Contains(blockIndex, i) {
			layoutDetResults = append(layoutDetResults, blocks[i])
		}
	}
	return layoutDetResults
}
