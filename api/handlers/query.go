package handlers

import (
	"rag-api/models"
	"rag-api/services"

	"github.com/gofiber/fiber/v2"
)

type QueryHandler struct {
	ragService *services.RAGService
}

func NewQueryHandler(ragService *services.RAGService) *QueryHandler {
	return &QueryHandler{
		ragService: ragService,
	}
}

func (h *QueryHandler) Handle(c *fiber.Ctx) error {
	var req models.QueryRequest
	if err := c.BodyParser(&req); err != nil {
		return fiber.NewError(fiber.StatusBadRequest, "Invalid request body")
	}

	if req.Query == "" {
		return fiber.NewError(fiber.StatusBadRequest, "Query cannot be empty")
	}

	ctx := c.Context()

	answer, err := h.ragService.Query(ctx, req.Query)
	if err != nil {
		return err
	}

	return c.JSON(models.QueryResponse{Response: answer})
}
