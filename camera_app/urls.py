'''from django.urls import path
from . import views

urlpatterns = [
    path('', views.homepage, name='homepage'),
    path('brand-detection/', views.brand_detection, name='brand_detection'),
    path("brand-detection/stream/", views.brand_detection_stream, name="brand-detection-stream"),
    path("brand-detection/start/", views.start_stream, name="start-stream"),
    path("brand-detection/stop/", views.stop_stream, name="stop-stream"),
    path('get-brand-counts/', views.get_brand_counts, name='get-brand-counts'),
    path('expiry-detection/', views.expiry_detection, name='expiry_detection'),
    path('item-counting/', views.item_counting, name='item_counting'),
    path('freshness-detection/', views.freshness_detection, name='freshness_detection'),
]
'''

from django.urls import path
from . import views

urlpatterns = [
    path('', views.homepage, name='homepage'),
    path('brand-detection/', views.brand_detection, name='brand_detection'),
    path("brand-detection/stream/", views.brand_detection_stream, name="brand-detection-stream"),
    path("brand-detection/start/", views.start_stream, name="start-stream"),
    path("brand-detection/stop/", views.stop_stream, name="stop-stream"),
    path('get-brand-counts/', views.get_brand_counts, name='get-brand-counts'),
    

    path('item-counting/', views.item_counting, name='item_counting'),
    

    path('expiry-detection/', views.expiry_detection, name='expiry_detection'),
    path("expiry-detection/stream/", views.expiry_detection_stream, name="expiry-detection-stream"),
    path("expiry-detection/start/", views.start_stream, name="start-stream"),
    path("expiry-detection/stop/", views.stop_stream, name="stop-stream"),


    path('freshness-detection/', views.freshness_detection, name='freshness_detection'),
    path("fruit-detection/stream/", views.fruit_detection_stream, name="fruit-detection-stream"),
    path("fruit-detection/start/", views.start_fruit_stream, name="start-fruit-stream"),
    path("fruit-detection/stop/", views.stop_fruit_stream, name="stop-fruit-stream"),
    path('get-fruit-counts/', views.get_fruit_counts, name='get-fruit-counts'),


]
