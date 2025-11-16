package services

import (
	"context"
	"fmt"
	"strings"
)

const promptTemplate = `<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context:
%s
<|eot_id|><|start_header_id|>user<|end_header_id|>
Question:
%s
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
`

type RAGService struct {
	tei      *TEIService
	weaviate *WeaviateService
	vllm     *VLLMService
}

func NewRAGService(tei *TEIService, weaviate *WeaviateService, vllm *VLLMService) *RAGService {
	return &RAGService{
		tei:      tei,
		weaviate: weaviate,
		vllm:     vllm,
	}
}

func (s *RAGService) Query(ctx context.Context, query string) (string, error) {
	embedding, err := s.tei.GetEmbedding(ctx, query)
	if err != nil {
		return "", fmt.Errorf("failed to get embedding: %w", err)
	}

	contextChunks, err := s.weaviate.SearchSimilar(ctx, embedding)
	if err != nil {
		return "", fmt.Errorf("failed to search Weaviate: %w", err)
	}

	contextStr := buildContext(contextChunks)
	finalPrompt := fmt.Sprintf(promptTemplate, contextStr, query)

	answer, err := s.vllm.GenerateResponse(ctx, finalPrompt)
	if err != nil {
		return "", fmt.Errorf("failed to generate answer: %w", err)
	}

	return answer, nil
}

func buildContext(chunks []string) string {
	if len(chunks) == 0 {
		return "No relevant context found."
	}

	var builder strings.Builder
	for i, chunk := range chunks {
		builder.WriteString(fmt.Sprintf("--- Context Chunk %d ---\n%s\n\n", i+1, chunk))
	}
	return builder.String()
}
