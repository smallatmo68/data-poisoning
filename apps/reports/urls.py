from django.urls import path
from .views import CreateLLMReportView, CreateReportView, DownloadReportView, ReportListView

urlpatterns = [
    path('', CreateReportView.as_view(), name='report-create'),
    path('list/', ReportListView.as_view(), name='report-list'),
    path('generate-llm/', CreateLLMReportView.as_view(), name='report-llm'),
    path('<uuid:report_id>/download/', DownloadReportView.as_view(), name='report-download'),
]
