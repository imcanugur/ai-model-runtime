package http_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	internalHTTP "github.com/imcanugur/ai-model-runtime/internal/http"
	storeMemory "github.com/imcanugur/ai-model-runtime/internal/jobstore/memory"
	queueMemory "github.com/imcanugur/ai-model-runtime/internal/queue/memory"
	"github.com/imcanugur/ai-model-runtime/internal/service"
)

func TestHTTP_SwaggerDocs(t *testing.T) {
	// Verify JSON syntax of OpenAPI spec
	if err := internalHTTP.ValidateOpenAPISpec(); err != nil {
		t.Fatalf("OpenAPISpecJSON is invalid JSON: %v", err)
	}

	reg := &mockRegistry{}
	store := storeMemory.NewMemoryJobStore()
	q := queueMemory.NewMemoryQueue()

	svc, err := service.NewInferenceService(service.ServiceConfig{
		Registry: reg,
		JobStore: store,
		Queue:    q,
	})
	if err != nil {
		t.Fatalf("failed to create inference service: %v", err)
	}

	cfg := internalHTTP.DefaultServerConfig()
	router := internalHTTP.NewRouter(svc, cfg)
	ts := httptest.NewServer(router)
	defer ts.Close()

	// 1. Test GET /swagger redirect
	req, _ := http.NewRequest(http.MethodGet, ts.URL+"/swagger", nil)
	client := &http.Client{
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse // Don't follow redirect automatically
		},
	}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("GET /swagger failed: %v", err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusMovedPermanently {
		t.Errorf("expected 301 Moved Permanently on /swagger, got %d", resp.StatusCode)
	}
	if loc := resp.Header.Get("Location"); loc != "/swagger/" {
		t.Errorf("expected Location '/swagger/', got %q", loc)
	}

	// 2. Test GET /swagger/ (Swagger UI HTML)
	respUI, err := http.Get(ts.URL + "/swagger/")
	if err != nil {
		t.Fatalf("GET /swagger/ failed: %v", err)
	}
	defer func() { _ = respUI.Body.Close() }()
	if respUI.StatusCode != http.StatusOK {
		t.Errorf("expected 200 OK on /swagger/, got %d", respUI.StatusCode)
	}
	ct := respUI.Header.Get("Content-Type")
	if !strings.Contains(ct, "text/html") {
		t.Errorf("expected text/html content-type, got %q", ct)
	}
	htmlBody, _ := io.ReadAll(respUI.Body)
	if !strings.Contains(string(htmlBody), "SwaggerUIBundle") {
		t.Errorf("expected SwaggerUIBundle in HTML body")
	}

	// 3. Test GET /swagger/openapi.json
	respJSON, err := http.Get(ts.URL + "/swagger/openapi.json")
	if err != nil {
		t.Fatalf("GET /swagger/openapi.json failed: %v", err)
	}
	defer func() { _ = respJSON.Body.Close() }()
	if respJSON.StatusCode != http.StatusOK {
		t.Errorf("expected 200 OK on /swagger/openapi.json, got %d", respJSON.StatusCode)
	}
	jsonBytes, _ := io.ReadAll(respJSON.Body)
	var spec map[string]any
	if err := json.Unmarshal(jsonBytes, &spec); err != nil {
		t.Fatalf("failed to decode OpenAPI JSON: %v", err)
	}

	// Verify essential OpenAPI elements
	if spec["openapi"] != "3.0.3" {
		t.Errorf("expected openapi version 3.0.3, got %v", spec["openapi"])
	}

	paths, ok := spec["paths"].(map[string]any)
	if !ok {
		t.Fatalf("paths object missing or invalid")
	}

	// Verify dynamic examples extension
	examples, hasExamples := spec["x-runtime-examples"].([]any)
	if !hasExamples || len(examples) == 0 {
		t.Fatalf("expected x-runtime-examples to contain dynamic example definitions")
	}

	requiredEndpoints := []string{
		"/health",
		"/ready",
		"/v1/models",
		"/v1/models/{modelID}",
		"/v1/models/{modelID}/predict",
		"/v1/models/{modelID}/versions/{version}/predict",
		"/v1/jobs",
		"/v1/jobs/{jobID}",
		"/predict",
	}

	for _, ep := range requiredEndpoints {
		if _, exists := paths[ep]; !exists {
			t.Errorf("expected OpenAPI spec to contain path %q", ep)
		}
	}

	// 4. Test GET /openapi.json alias
	respAlias, err := http.Get(ts.URL + "/openapi.json")
	if err != nil {
		t.Fatalf("GET /openapi.json failed: %v", err)
	}
	defer func() { _ = respAlias.Body.Close() }()
	if respAlias.StatusCode != http.StatusOK {
		t.Errorf("expected 200 OK on /openapi.json alias, got %d", respAlias.StatusCode)
	}

	// 5. Test GET /v1/models/{modelID}
	respModel, err := http.Get(ts.URL + "/v1/models/text-generation-tr")
	if err != nil {
		t.Fatalf("GET /v1/models/text-generation-tr failed: %v", err)
	}
	defer func() { _ = respModel.Body.Close() }()
	if respModel.StatusCode != http.StatusOK {
		t.Errorf("expected 200 OK on /v1/models/text-generation-tr, got %d", respModel.StatusCode)
	}

	// 6. Test GET /v1/models/{modelID} Not Found
	respNotFound, err := http.Get(ts.URL + "/v1/models/non-existent-model")
	if err != nil {
		t.Fatalf("GET /v1/models/non-existent-model failed: %v", err)
	}
	defer func() { _ = respNotFound.Body.Close() }()
	if respNotFound.StatusCode != http.StatusNotFound {
		t.Errorf("expected 404 Not Found on non-existent model, got %d", respNotFound.StatusCode)
	}
}
