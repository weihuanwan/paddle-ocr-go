package vl

import (
	"encoding/base64"
	"math"

	"github.com/weihuanwan/paddleocr-go/common"
	"github.com/weihuanwan/paddleocr-go/layout"
	"gocv.io/x/gocv"
)

func matToBase64(mat *gocv.Mat) (string, error) {

	buf, err := gocv.IMEncode(".png", *mat)
	if err != nil {
		return "", err
	}
	defer buf.Close()
	// 2. 转 base64
	base64Str := "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.GetBytes())
	return base64Str, nil
}
func filterOverlapBoxes(results []*common.LayoutDetResult, layoutShapeMode string) []*common.LayoutDetResult {

	// 1️⃣ 过滤掉 reference（和 Python 一样）
	boxes := make([]*common.LayoutDetResult, 0)
	for _, r := range results {
		if r.Label != "reference" {
			boxes = append(boxes, r)
		}
	}

	// 2️⃣ set
	dropped := make(map[int]struct{})

	for i := 0; i < len(boxes); i++ {

		pointI := boxes[i].Point
		x1, y1, x2, y2 := pointI[0], pointI[1], pointI[2], pointI[3]
		w, h := x2-x1, y2-y1

		// Python: 直接 add，但不 continue
		if w < 6 || h < 6 {
			dropped[i] = struct{}{}
		}

		for j := i + 1; j < len(boxes); j++ {

			if _, ok := dropped[i]; ok {
				continue
			}
			if _, ok := dropped[j]; ok {
				continue
			}

			overlapRatio := calculateOverlapRatio(pointI, boxes[j].Point, "small")

			labelI := boxes[i].Label
			labelJ := boxes[j].Label

			// inline_formula
			if labelI == "inline_formula" || labelJ == "inline_formula" {

				if overlapRatio > 0.5 {
					if labelI == "inline_formula" {
						dropped[i] = struct{}{}
					}
					if labelJ == "inline_formula" {
						dropped[j] = struct{}{}
					}
					continue
				}
			}

			if overlapRatio > 0.7 {

				// polygon 判断
				if layoutShapeMode != "rect" && boxes[i].PolygonPoints != nil {

					polyOverlapRatio := layout.CalculatePolygonOverlapRatio(
						boxes[i].PolygonPoints,
						boxes[j].PolygonPoints,
						"small",
					)

					if polyOverlapRatio < 0.7 {
						continue
					}
				}

				boxAreaI := calculateArea(pointI)
				boxAreaJ := calculateArea(boxes[j].Point)

				// ===== 关键：Python labels 逻辑 =====
				labels := map[string]struct{}{
					labelI: {},
					labelJ: {},
				}

				_, hasImage := labels["image"]
				_, hasTable := labels["table"]
				_, hasSeal := labels["seal"]
				_, hasChart := labels["chart"]

				if (hasImage || hasTable || hasSeal || hasChart) && len(labels) > 1 {

					// Python:
					// if "table" not in labels or labels <= {...}
					if !hasTable || isSubset(labels, map[string]struct{}{
						"table": {},
						"image": {},
						"seal":  {},
						"chart": {},
					}) {
						continue
					}
				}

				if boxAreaI >= boxAreaJ {
					dropped[j] = struct{}{}
				} else {
					dropped[i] = struct{}{}
				}
			}
		}
	}

	// 3️⃣ 过滤
	filtered := make([]*common.LayoutDetResult, 0, len(boxes))
	for i, b := range boxes {
		if _, ok := dropped[i]; !ok {
			filtered = append(filtered, b)
		}
	}

	return filtered
}
func isSubset(a, b map[string]struct{}) bool {
	for k := range a {
		if _, ok := b[k]; !ok {
			return false
		}
	}
	return true
}

// 计算面积
func calculateArea(p []int) float64 {
	if len(p) != 4 {
		return 0
	}
	width := math.Max(0, float64(p[2]-p[0]))
	height := math.Max(0, float64(p[3]-p[1]))
	return width * height
}

// 计算重叠比例
func calculateOverlapRatio(point1 []int, point2 []int, mode string) float64 {

	if len(point1) != 4 || len(point2) != 4 {
		return 0
	}

	// 交集区域
	xMinInter := math.Max(float64(point1[0]), float64(point2[0]))
	yMinInter := math.Max(float64(point1[1]), float64(point2[1]))
	xMaxInter := math.Min(float64(point1[2]), float64(point2[2]))
	yMaxInter := math.Min(float64(point1[3]), float64(point2[3]))

	// 宽高（防止负数）
	interWidth := math.Max(0, xMaxInter-xMinInter)
	interHeight := math.Max(0, yMaxInter-yMinInter)

	interArea := interWidth * interHeight

	// 各自面积
	area1 := calculateArea(point1)
	area2 := calculateArea(point2)

	var refArea float64

	switch mode {
	case "union":
		refArea = area1 + area2 - interArea
	case "small":
		refArea = math.Min(area1, area2)
	case "large":
		refArea = math.Max(area1, area2)
	default:
		return 0
	}
	if refArea == 0 {
		return 0
	}

	return interArea / refArea
}
