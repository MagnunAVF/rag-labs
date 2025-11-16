package services

import (
	"context"
	"errors"
	"fmt"

	"github.com/sashabaranov/go-openai"
)

type VLLMService struct {
	client    *openai.Client
	modelName string
	maxTokens int
}

func NewVLLMService(client *openai.Client, modelName string, maxTokens int) *VLLMService {
	return &VLLMService{
		client:    client,
		modelName: modelName,
		maxTokens: maxTokens,
	}
}

func (s *VLLMService) GenerateResponse(ctx context.Context, prompt string) (string, error) {
	req := openai.ChatCompletionRequest{
		Model: s.modelName,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleUser,
				Content: prompt,
			},
		},
		MaxTokens: s.maxTokens,
		Stop:      []string{"<|eot_id|>"},
	}

	resp, err := s.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", fmt.Errorf("vLLM completion failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", errors.New("no completion choices returned from vLLM")
	}

	return resp.Choices[0].Message.Content, nil
}
