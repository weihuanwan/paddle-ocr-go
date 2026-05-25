package vl

/*
*
公共的vl 请求参数和返回参数
*/
type ChatCompletionRequest struct {
	Model             string     `json:"model"`
	Messages          []Messages `json:"messages"`
	Temperature       float32    `json:"temperature"`
	TopK              int        `json:"top_k"`
	SkipSpecialTokens bool       `json:"skip_special_tokens"`
}

type Messages struct {
	Role    string    `json:"role"`
	Content []Content `json:"content"`
}

type Content struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

type ImageURL struct {
	URL string `json:"url"`
}

type ChatCompletionResponse struct {
	ID                string      `json:"id"`
	Object            string      `json:"object"`
	Created           int64       `json:"created"`
	Model             string      `json:"model"`
	Choices           []Choice    `json:"choices"`
	ServiceTier       string      `json:"service_tier"`
	SystemFingerprint string      `json:"system_fingerprint"`
	Usage             Usage       `json:"usage"`
	PromptLogprobs    interface{} `json:"prompt_logprobs"`
	PromptTokenIDs    interface{} `json:"prompt_token_ids"`
	KVTransferParams  interface{} `json:"kv_transfer_params"`
}

type Choice struct {
	Index        int         `json:"index"`
	Message      Message     `json:"message"`
	Logprobs     interface{} `json:"logprobs"`
	FinishReason string      `json:"finish_reason"`
	StopReason   string      `json:"stop_reason"`
	TokenIDs     interface{} `json:"token_ids"`
}

type Message struct {
	Role         string        `json:"role"`
	Content      string        `json:"content"`
	Refusal      string        `json:"refusal"`
	Annotations  interface{}   `json:"annotations"`
	Audio        interface{}   `json:"audio"`
	FunctionCall interface{}   `json:"function_call"`
	ToolCalls    []interface{} `json:"tool_calls"`
	Reasoning    interface{}   `json:"reasoning"`
}

type Usage struct {
	PromptTokens        int         `json:"prompt_tokens"`
	TotalTokens         int         `json:"total_tokens"`
	CompletionTokens    int         `json:"completion_tokens"`
	PromptTokensDetails interface{} `json:"prompt_tokens_details"`
}
