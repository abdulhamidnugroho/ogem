package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yanolja/ogem/openai"
)

func TestCreateEmbeddings(t *testing.T) {
	mockResponse := openai.EmbeddingResponse{
		Data: []openai.Data{
			{
				Embedding: []float32{0.1, 0.2, 0.3},
			},
		},
		Model: "text-embedding-ada-002",
	}

	responseBody, _ := json.Marshal(mockResponse)

	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "POST", r.Method)
		require.Equal(t, "/embeddings", r.URL.Path)
		require.Equal(t, "application/json", r.Header.Get("Content-Type"))
		require.Contains(t, r.Header.Get("Authorization"), "Bearer")

		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		require.NotEmpty(t, body)

		if r.Header.Get("Authorization") != "Bearer test-api-key" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}

		w.WriteHeader(http.StatusOK)
		w.Write(responseBody)
	}))
	defer testServer.Close()

	baseURL, err := url.Parse(testServer.URL)
	require.NoError(t, err)

	endpoint := &Endpoint{
		apiKey:  "test-api-key",
		baseUrl: baseURL,
		client:  testServer.Client(),
	}

	t.Run("Valid embedding request", func(t *testing.T) {
		embeddingRequest := &openai.EmbeddingRequest{
			Model: "text-embedding-ada-002",
			Input: "test input",
		}

		ctx := context.Background()
		response, err := endpoint.CreateEmbeddings(ctx, embeddingRequest)

		assert.NoError(t, err)
		assert.NotNil(t, response)
		assert.Equal(t, "text-embedding-ada-002", response.Model)
		assert.Equal(t, []float32{0.1, 0.2, 0.3}, response.Data[0].Embedding)
	})

	t.Run("Unauthorized request", func(t *testing.T) {
		endpoint.apiKey = "wrong-key"

		embeddingRequest := &openai.EmbeddingRequest{
			Model: "text-embedding-ada-002",
			Input: "test input",
		}

		ctx := context.Background()
		response, err := endpoint.CreateEmbeddings(ctx, embeddingRequest)

		assert.Error(t, err)
		assert.Nil(t, response)
	})
}

func TestCreateEmbeddingsWithInvalidRequest(t *testing.T) {
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "POST", r.Method)
		require.Equal(t, "/embeddings", r.URL.Path)
		require.Equal(t, "application/json", r.Header.Get("Content-Type"))
		require.Contains(t, r.Header.Get("Authorization"), "Bearer")

		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		require.NotEmpty(t, body)

		if r.Header.Get("Authorization") != "Bearer test-api-key" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}

		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error": {
				"message": "Invalid value for 'encoding_format' = string. Supported values: ['float', 'base64'].", 
				"type": "invalid_request_error", "param": null, "code": null}
				}`))
	}))
	defer testServer.Close()

	baseURL, err := url.Parse(testServer.URL)
	require.NoError(t, err)

	endpoint := &Endpoint{
		apiKey:  "test-api-key",
		baseUrl: baseURL,
		client:  testServer.Client(),
	}

	t.Run("Invalid embedding request", func(t *testing.T) {
		embeddingRequest := &openai.EmbeddingRequest{
			Model:          "text-embedding-ada-002",
			Input:          "test input",
			EncodingFormat: "string",
		}

		ctx := context.Background()
		response, err := endpoint.CreateEmbeddings(ctx, embeddingRequest)

		assert.Error(t, err)
		assert.Nil(t, response)
		assert.Contains(t, err.Error(), "Invalid value for 'encoding_format' = string")
	})
}

func TestCreateEmbeddingsRateLimitExceeded(t *testing.T) {
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "POST", r.Method)
		require.Equal(t, "/embeddings", r.URL.Path)
		require.Equal(t, "application/json", r.Header.Get("Content-Type"))
		require.Contains(t, r.Header.Get("Authorization"), "Bearer")

		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte(`{"error": {"message": "Rate limit exceeded", "type": "rate_limit_exceeded", "param": null, "code": null}}`))
	}))
	defer testServer.Close()

	baseURL, err := url.Parse(testServer.URL)
	require.NoError(t, err)

	endpoint := &Endpoint{
		apiKey:  "test-api-key",
		baseUrl: baseURL,
		client:  testServer.Client(),
	}

	t.Run("Rate limit exceeded", func(t *testing.T) {
		embeddingRequest := &openai.EmbeddingRequest{
			Model: "text-embedding-ada-002",
			Input: "test input",
		}

		ctx := context.Background()
		response, err := endpoint.CreateEmbeddings(ctx, embeddingRequest)

		assert.Error(t, err)
		assert.Nil(t, response)
		assert.Contains(t, err.Error(), "Rate limit exceeded")
	})
}
