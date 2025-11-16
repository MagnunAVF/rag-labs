package services

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"rag-api/models"
)

type TEIService struct {
	baseURL    string
	httpClient *http.Client
}

func NewTEIService(baseURL string, httpClient *http.Client) *TEIService {
	return &TEIService{
		baseURL:    baseURL,
		httpClient: httpClient,
	}
}

func (s *TEIService) GetEmbedding(ctx context.Context, text string) ([]float64, error) {
	payload := models.TEIRequest{
		Inputs:    []string{text},
		Truncate:  true,
		Normalize: true,
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal TEI request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		s.baseURL+"/embed", bytes.NewBuffer(payloadBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create TEI request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("TEI request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("TEI server returned status %d", resp.StatusCode)
	}

	var response models.TEIResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode TEI response: %w", err)
	}

	if len(response) == 0 || len(response[0]) == 0 {
		return nil, errors.New("empty embedding response from TEI")
	}

	return response[0], nil
}
