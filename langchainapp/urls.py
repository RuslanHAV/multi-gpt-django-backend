from django.urls import path
from langchainapp.views import *

urlpatterns = [
    path('url_embedding', EmbeddingURL.as_view(), name="url_embedding"),
    path('pdf_embedding', EmbeddingPDF.as_view(), name="pdf_embedding"),
    path('chat', CHAT.as_view(), name="chat"),
]
