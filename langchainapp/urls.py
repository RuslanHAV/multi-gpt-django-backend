from django.urls import path
from langchainapp.views import *

urlpatterns = [
    path('url_embedding', Embedding.as_view(), name="url_embedding"),
    path('chat', CHAT.as_view(), name="chat"),
]
