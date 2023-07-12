from django.urls import path
from langchainapp.views import *

urlpatterns = [
    path('url_embedding', EmbeddingURL.as_view(), name="url_embedding"),
    path('pdf_embedding', EmbeddingPDF.as_view(), name="pdf_embedding"),
    path('save_attr', LangAttr.as_view(), name="save_attr"),
    path('get_attr', GetLangAttr.as_view(), name="get_attr"),
    path('chat', CHAT.as_view(), name="chat"),
]
