package models

type QueryRequest struct {
	Query string `json:"query" validate:"required,min=1"`
}

type QueryResponse struct {
	Response string `json:"response"`
}

type ErrorResponse struct {
	Error string `json:"error"`
}

type TEIRequest struct {
	Inputs    []string `json:"inputs"`
	Truncate  bool     `json:"truncate"`
	Normalize bool     `json:"normalize"`
}

type TEIResponse [][]float64
