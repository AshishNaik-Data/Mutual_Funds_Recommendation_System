"""FastAPI app. Run with:  uvicorn mf_recommender.api:app --port 8000  (from backend/)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings
from .data import RISK_BANDS
from .llm import describe_model
from .schemas import CategoriesResponse, RecommendRequest, RecommendResponse
from .service import RecommenderService, UnknownCategoryError

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, service: RecommenderService | None = None) -> FastAPI:
    settings = settings or (service.settings if service else Settings.from_env())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Models load once at startup, not on import and not per request.
        if getattr(app.state, "service", None) is None:
            app.state.service = RecommenderService.from_settings(settings)
        yield

    app = FastAPI(title="Mutual Fund Recommender", version="2.0.0", lifespan=lifespan)
    app.state.service = service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,  # "*" origins with credentials is rejected by browsers
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def get_service(request: Request) -> RecommenderService:
        svc = request.app.state.service
        if svc is None:
            raise HTTPException(503, "Service is still starting")
        return svc

    @app.get("/health")
    def health(request: Request) -> dict:
        svc = get_service(request)
        return {
            "status": "ok",
            "funds": len(svc.funds),
            # llm_backend, base_model, adapter, fine_tuned: what writes the explanations
            **describe_model(svc.settings, svc.llm is not None),
            "embedding_model": svc.settings.embedding_model,
            "live_nav": svc.nav_client is not None,
        }

    @app.get("/categories", response_model=CategoriesResponse)
    def categories(request: Request) -> CategoriesResponse:
        return CategoriesResponse(categories=get_service(request).categories, risk_levels=list(RISK_BANDS))

    # Plain `def`: FastAPI runs it in a worker thread, so slow model calls don't block the event loop.
    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(body: RecommendRequest, request: Request) -> RecommendResponse:
        svc = get_service(request)
        try:
            result = svc.recommend(body.goal, body.category, body.risk, top_k=body.top_k, explain=body.explain)
        except UnknownCategoryError as exc:
            raise HTTPException(422, f"{exc}. Valid categories: GET /categories") from exc
        return RecommendResponse(funds=result.funds, query=result.query, notice=result.notice, filters=result.filters)

    return app


app = create_app()
