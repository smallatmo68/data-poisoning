const { createApp, ref, reactive, computed, onMounted, watch, onUnmounted } = Vue;

// ── API 工具 ──────────────────────────────────────────────────────
const TOKEN_KEY = 'dpds_access_token';
function getToken() { return localStorage.getItem(TOKEN_KEY) || ''; }
function setToken(t) { localStorage.setItem(TOKEN_KEY, t); }
function clearToken() { localStorage.removeItem(TOKEN_KEY); }

async function api(method, path, body = null, isForm = false) {
    const headers = {};
    const tok = getToken();
    if (tok) headers['Authorization'] = `Bearer ${tok}`;
    if (!isForm && body) headers['Content-Type'] = 'application/json';
    const opts = { method, headers };
    if (body) opts.body = isForm ? body : JSON.stringify(body);
    const res = await fetch(path, opts);
    const data = await res.json().catch(() => ({ code: res.status, msg: `HTTP ${res.status}` }));
    if (res.status === 401 && tok) { clearToken(); window.location.reload(); }
    return data;
}
const GET  = (p)       => api('GET', p);
const POST = (p, b, f) => api('POST', p, b, f);
const PUT  = (p, b)    => api('PUT', p, b);
const DEL  = (p)       => api('DELETE', p);

// ── 主应用 ──────────────────────────────────────────────────────
const app = createApp({
    setup() {
        const token   = ref(getToken());
        const user    = ref(null);
        const page    = ref('dashboard');
        const loading = reactive({
            auth: false, datasets: false, tasks: false, methods: false,
            upload: false, submit: false, preview: false, defense: false,
            report: false, detectors: false, preprocess: false,
            adminUsers: false, logs: false, dashboard: false,
        });

        // 核心数据
        const datasets    = ref([]);
        const tasks       = ref([]);
        const methods     = ref([]);
        const detectors   = ref([]);
        const currentTask = ref(null);
        const pollTimer   = ref(null);

        // 登录
        const loginForm = reactive({ username: '', password: '' });
        const loginErr  = ref('');

        // 提交检测
        const step       = ref(0);
        const uploadFile = ref(null);
        const uploadedId = ref(null);
        const submitForm = reactive({
            sourceType: 'upload', datasetName: '', existingId: null,
            detectors: ['cleanlab', 'isolation_forest'], baselineId: null,
        });

        // 数据集
        const datasetDetail     = ref(null);
        const showDatasetDetail = ref(false);
        const showUploadDialog  = ref(false);
        const dsUploadFile      = ref(null);
        const dsUploadForm      = reactive({ datasetName: '', datasetType: 'csv' });
        const datasetPreviewRows  = ref([]);
        const datasetPreviewCols  = ref([]);
        const datasetPreviewTotal = ref(0);

        // 预处理
        const preprocessForm = reactive({
            datasetId: null, dedup: true, missingStrategy: 'drop',
            normalize: true, encodeCategorical: true,
        });
        const preprocessResult = ref(null);

        // 防御
        const defenseResults  = ref(null);
        const defenseStrategy = reactive({ default: 'remove', relabel: {}, ignore: [] });

        // 报告
        const reportInfo   = ref(null);
        const analysisData = ref(null);

        // Dashboard
        const dashStats = reactive({
            total_datasets: 0, total_tasks: 0, total_reports: 0,
            high_risk: 0, medium_risk: 0, low_risk: 0,
        });

        // Admin
        const adminStats  = reactive({
            total_users: 0, total_datasets: 0, total_tasks: 0, total_reports: 0,
            recent_tasks: [], recent_logs: [],
        });
        const adminUsers = ref([]);

        // 审计日志
        const auditLogs     = ref([]);
        const auditLogTotal = ref(0);
        const auditLogPage  = ref(1);
        const auditPageSize = 20;
        const logFilter     = reactive({ action: '', user: '' });

        // ── 计算属性 ──────────────────────────────────────────────
        const isLoggedIn     = computed(() => !!token.value);
        const completedTasks = computed(() => tasks.value.filter(t => t.status === 'success'));

        const pageTitle = computed(() => ({
            dashboard: '工作台', submit: '提交检测', tasks: '检测任务',
            reports: '检测报告', datasets: '数据集管理', methods: '检测算法',
            defense: '数据无害化', preprocess: '数据预处理',
            'report-center': '报告中心', admin: '系统概览',
            users: '用户管理', logs: '审计日志',
        }[page.value] || ''));

        const datasetColumns = computed(() => {
            if (!datasetDetail.value?.column_meta) return [];
            return Object.entries(datasetDetail.value.column_meta)
                .map(([name, info]) => ({ name, ...info }));
        });

        const currentDataset = computed(() =>
            datasets.value.find(d => d.id === currentTask.value?.dataset?.id) || null
        );

        const selectedRequiresBaseline = computed(() =>
            detectors.value.some(d => submitForm.detectors.includes(d.detector_name) && d.requires_baseline)
        );

        const reportStats = computed(() => {
            const results = currentTask.value?.results;
            if (!results?.length) return null;
            const total = results.length;
            const detMap = {}, rtMap = {}, sugCount = {};
            const confBuckets = [0, 0, 0, 0, 0];
            for (const r of results) {
                if (!detMap[r.detector_name]) detMap[r.detector_name] = { count: 0, confSum: 0, maxConf: 0 };
                detMap[r.detector_name].count++;
                detMap[r.detector_name].confSum += r.confidence;
                detMap[r.detector_name].maxConf = Math.max(detMap[r.detector_name].maxConf, r.confidence);
                rtMap[r.risk_type] = (rtMap[r.risk_type] || 0) + 1;
                sugCount[r.suggestion] = (sugCount[r.suggestion] || 0) + 1;
                confBuckets[Math.min(Math.floor(r.confidence * 5), 4)]++;
            }
            const byDetector = Object.entries(detMap).map(([name, v]) => ({
                name, count: v.count, avgConf: +((v.confSum / v.count) * 100).toFixed(1),
                maxConf: +(v.maxConf * 100).toFixed(1),
            })).sort((a, b) => b.count - a.count);
            const byRiskType = Object.entries(rtMap).map(([type, count]) => ({
                type, count, pct: +((count / total) * 100).toFixed(1),
            })).sort((a, b) => b.count - a.count);
            const sugList = Object.entries(sugCount).map(([sug, count]) => ({
                sug, count, pct: +((count / total) * 100).toFixed(1),
            })).sort((a, b) => b.count - a.count);
            return {
                total, byDetector, byRiskType, sugList, sugCount, confBuckets,
                maxBucket: Math.max(...confBuckets, 1),
                maxDetCount: Math.max(...byDetector.map(d => d.count), 1),
                maxRtCount: Math.max(...byRiskType.map(r => r.count), 1),
            };
        });

        const conclusionText = computed(() => {
            const task = currentTask.value;
            if (!task) return '';
            const score = task.risk_score;
            const n = task.total_suspicious || 0;
            const totalSamples = currentDataset.value?.sample_count;
            const pct = totalSamples ? ((n / totalSamples) * 100).toFixed(1) + '%' : '—';
            if (score >= 0.15)
                return `该数据集存在高度数据投毒风险（综合风险分数 ${(score*100).toFixed(1)}%），共发现 ${n} 个可疑样本（约占 ${pct}）。强烈建议在训练前对数据集进行全面清洗。`;
            if (score >= 0.05)
                return `该数据集存在中等数据投毒风险（综合风险分数 ${(score*100).toFixed(1)}%），共发现 ${n} 个可疑样本（约占 ${pct}）。建议针对高置信度样本进行人工复核。`;
            if (score != null)
                return `该数据集安全风险较低（综合风险分数 ${(score*100).toFixed(1)}%），发现 ${n} 个潜在可疑样本（约占 ${pct}）。整体数据质量可信。`;
            return '检测任务执行异常，请查看下方错误信息。';
        });

        // ── 登录/注销 ─────────────────────────────────────────────
        async function login() {
            loginErr.value = '';
            loading.auth = true;
            const r = await POST('/api/auth/login/', loginForm);
            loading.auth = false;
            if (r.code !== 0) { loginErr.value = r.msg || '登录失败'; return; }
            setToken(r.data.access);
            token.value = r.data.access;
            user.value  = r.data.user;
            loadAll();
        }
        function logout() { clearToken(); token.value = ''; user.value = null; page.value = 'dashboard'; }

        // ── 数据加载 ──────────────────────────────────────────────
        async function loadAll() {
            loadDatasets(); loadTasks(); loadMethods(); loadDetectors(); loadDashboard();
            if (!user.value) {
                const r = await GET('/api/auth/profile/');
                if (r.code === 0) user.value = r.data;
            }
            if (user.value?.is_staff) { loadAdminDashboard(); loadAdminUsers(); }
        }

        async function loadDashboard() {
            const r = await GET('/api/detection/tasks/');
            if (r.code === 0) {
                tasks.value = r.data || [];
                const allTasks = r.data || [];
                dashStats.total_tasks = allTasks.length;
                dashStats.high_risk = allTasks.filter(t => t.risk_score >= 0.15).length;
                dashStats.medium_risk = allTasks.filter(t => t.risk_score >= 0.05 && t.risk_score < 0.15).length;
                dashStats.low_risk = allTasks.filter(t => t.risk_score != null && t.risk_score < 0.05).length;
            }
            const rd = await GET('/api/datasets/?page_size=1000');
            if (rd.code === 0) {
                const d = rd.data;
                const list = Array.isArray(d) ? d : (d.results || []);
                dashStats.total_datasets = list.length;
            }
        }

        async function loadDatasets() {
            loading.datasets = true;
            const r = await GET('/api/datasets/?page_size=200');
            loading.datasets = false;
            if (r.code === 0) {
                const d = r.data;
                datasets.value = Array.isArray(d) ? d : (d.results || []);
                dashStats.total_datasets = datasets.value.length;
            }
        }

        async function loadTasks() {
            loading.tasks = true;
            const r = await GET('/api/detection/tasks/');
            loading.tasks = false;
            if (r.code === 0) tasks.value = r.data || [];
        }

        async function loadMethods() {
            loading.methods = true;
            const r = await GET('/api/detection/detectors/');
            loading.methods = false;
            if (r.code === 0) methods.value = r.data || [];
        }

        async function loadDetectors() {
            loading.detectors = true;
            const r = await GET('/api/detection/detectors/');
            loading.detectors = false;
            if (r.code === 0) detectors.value = r.data || [];
        }

        async function loadAdminDashboard() {
            const r = await GET('/api/auth/admin/dashboard/');
            if (r.code === 0) Object.assign(adminStats, r.data);
        }

        async function loadAdminUsers() {
            loading.adminUsers = true;
            const r = await GET('/api/auth/admin/users/');
            loading.adminUsers = false;
            if (r.code === 0) adminUsers.value = r.data || [];
        }

        async function loadAuditLogs() {
            loading.logs = true;
            let url = `/api/auth/admin/audit-logs/?page=${auditLogPage.value}&page_size=${auditPageSize}`;
            if (logFilter.action) url += `&action=${logFilter.action}`;
            if (logFilter.user) url += `&user=${logFilter.user}`;
            const r = await GET(url);
            loading.logs = false;
            if (r.code === 0) {
                auditLogs.value = r.data.results || [];
                auditLogTotal.value = r.data.total || 0;
            }
        }

        async function toggleUser(u) {
            const action = u.is_active ? '禁用' : '启用';
            await ElementPlus.ElMessageBox.confirm(`确认${action}用户「${u.username}」？`, '提示', { type: 'warning' });
            const r = await POST(`/api/auth/admin/users/${u.id}/toggle/`);
            if (r.code === 0) { ElementPlus.ElMessage.success(r.msg); loadAdminUsers(); }
            else ElementPlus.ElMessage.error(r.msg);
        }

        // ── 数据集上传 ────────────────────────────────────────────
        function handleFileChange(file) {
            uploadFile.value = file.raw;
            if (!submitForm.datasetName) submitForm.datasetName = file.name.replace(/\.[^.]+$/, '');
        }

        async function doUpload() {
            if (!uploadFile.value) { ElementPlus.ElMessage.warning('请先选择文件'); return; }
            loading.upload = true;
            const fd = new FormData();
            fd.append('file', uploadFile.value);
            fd.append('dataset_name', submitForm.datasetName || uploadFile.value.name);
            fd.append('dataset_type', 'csv');
            const r = await POST('/api/datasets/upload/', fd, true);
            loading.upload = false;
            if (r.code !== 0) { ElementPlus.ElMessage.error(r.msg || '上传失败'); return; }
            uploadedId.value = r.data.dataset_id;
            if (r.data.duplicate) ElementPlus.ElMessage.warning('检测到重复文件，已使用已有数据集');
            else ElementPlus.ElMessage.success('上传成功');
            await loadDatasets();
            step.value = 1;
        }

        function useExistingDataset() {
            if (!submitForm.existingId) { ElementPlus.ElMessage.warning('请选择数据集'); return; }
            uploadedId.value = submitForm.existingId;
            step.value = 1;
        }

        function openUploadDialog() {
            dsUploadFile.value = null;
            dsUploadForm.datasetName = '';
            dsUploadForm.datasetType = 'csv';
            showUploadDialog.value = true;
        }

        function handleDsUploadChange(file) {
            dsUploadFile.value = file.raw;
            if (!dsUploadForm.datasetName) dsUploadForm.datasetName = file.name.replace(/\.[^.]+$/, '');
        }

        async function doDsUpload() {
            if (!dsUploadFile.value) { ElementPlus.ElMessage.warning('请先选择文件'); return; }
            loading.upload = true;
            const fd = new FormData();
            fd.append('file', dsUploadFile.value);
            fd.append('dataset_name', dsUploadForm.datasetName || dsUploadFile.value.name);
            fd.append('dataset_type', dsUploadForm.datasetType);
            const r = await POST('/api/datasets/upload/', fd, true);
            loading.upload = false;
            if (r.code !== 0) { ElementPlus.ElMessage.error(r.msg || '上传失败'); return; }
            if (r.data.duplicate) ElementPlus.ElMessage.warning('检测到重复文件，已使用已有数据集');
            else ElementPlus.ElMessage.success('数据集上传成功');
            showUploadDialog.value = false;
            await loadDatasets();
        }

        // ── 数据集详情 ────────────────────────────────────────────
        async function openDatasetDetail(ds) {
            datasetDetail.value = { ...ds };
            datasetPreviewRows.value = [];
            datasetPreviewCols.value = [];
            datasetPreviewTotal.value = 0;
            showDatasetDetail.value = true;
            loading.preview = true;
            const r = await GET(`/api/datasets/${ds.id}/preview/?page=1&page_size=20`);
            loading.preview = false;
            if (r.code === 0 && r.data) {
                datasetPreviewRows.value = r.data.rows || [];
                datasetPreviewCols.value = r.data.columns || [];
                datasetPreviewTotal.value = r.data.total || ds.sample_count || 0;
            }
        }

        function startDetectFromDs(ds) {
            submitForm.existingId = ds.id;
            submitForm.sourceType = 'existing';
            uploadedId.value = ds.id;
            step.value = 1;
            page.value = 'submit';
        }

        async function deleteDataset(ds) {
            await ElementPlus.ElMessageBox.confirm(`确认删除数据集「${ds.dataset_name}」？`, '提示', { type: 'warning' });
            const r = await DEL(`/api/datasets/${ds.id}/`);
            if (r.code === 0) { ElementPlus.ElMessage.success('已删除'); loadDatasets(); }
            else ElementPlus.ElMessage.error(r.msg);
        }

        function toggleDetector(name) {
            const idx = submitForm.detectors.indexOf(name);
            if (idx >= 0) submitForm.detectors.splice(idx, 1);
            else submitForm.detectors.push(name);
        }

        // ── 提交检测 ──────────────────────────────────────────────
        async function submitTask() {
            if (!uploadedId.value) { ElementPlus.ElMessage.warning('请先选择或上传数据集'); return; }
            if (!submitForm.detectors.length) { ElementPlus.ElMessage.warning('请至少选择一个检测算法'); return; }
            if (selectedRequiresBaseline.value && !submitForm.baselineId) {
                ElementPlus.ElMessage.warning('所选算法需要基准数据集，请在上一步选择'); return;
            }
            loading.submit = true;
            const payload = { dataset_id: uploadedId.value, detectors: submitForm.detectors };
            if (submitForm.baselineId) payload.baseline_dataset_id = submitForm.baselineId;
            const r = await POST('/api/detection/tasks/', payload);
            loading.submit = false;
            if (r.code !== 0) { ElementPlus.ElMessage.error(r.msg || '提交失败'); return; }
            ElementPlus.ElMessage.success('检测任务已创建，正在后台运行');
            step.value = 0; uploadFile.value = null; uploadedId.value = null;
            submitForm.datasetName = ''; submitForm.existingId = null;
            submitForm.baselineId = null; submitForm.detectors = ['cleanlab', 'isolation_forest'];
            await loadTasks(); page.value = 'tasks'; startPolling();
        }

        // ── 任务轮询 ──────────────────────────────────────────────
        function startPolling() {
            stopPolling();
            pollTimer.value = setInterval(async () => {
                const running = tasks.value.some(t => t.status === 'queued' || t.status === 'running');
                if (!running) { stopPolling(); return; }
                await loadTasks();
            }, 3000);
        }
        function stopPolling() { if (pollTimer.value) { clearInterval(pollTimer.value); pollTimer.value = null; } }
        onUnmounted(stopPolling);

        // ── 查看任务详情 ──────────────────────────────────────────
        async function viewTask(taskItem) {
            const r = await GET(`/api/detection/tasks/${taskItem.task_id}/`);
            if (r.code !== 0) { ElementPlus.ElMessage.error('获取任务详情失败'); return; }
            const detail = r.data;
            if (detail.status === 'success') {
                const rr = await GET(`/api/detection/tasks/${taskItem.task_id}/results/`);
                if (rr.code === 0) {
                    detail.results = rr.data.results || [];
                    detail.total_suspicious = rr.data.total_suspicious || 0;
                } else { detail.results = []; detail.total_suspicious = 0; }
            }
            currentTask.value = detail;
            defenseResults.value = null; reportInfo.value = null; analysisData.value = null;
            page.value = 'reports';
            if (detail.status === 'success') {
                const ra = await GET(`/api/detection/tasks/${taskItem.task_id}/analysis/`);
                if (ra.code === 0) analysisData.value = ra.data;
            }
        }

        function closeTask() {
            currentTask.value = null; defenseResults.value = null;
            reportInfo.value = null; analysisData.value = null;
        }

        // ── 预处理 ────────────────────────────────────────────────
        async function runPreprocess() {
            if (!preprocessForm.datasetId) { ElementPlus.ElMessage.warning('请选择数据集'); return; }
            loading.preprocess = true;
            preprocessResult.value = null;
            const r = await POST(`/api/datasets/${preprocessForm.datasetId}/preprocess/`, {
                params: {
                    dedup: preprocessForm.dedup,
                    missing_strategy: preprocessForm.missingStrategy,
                    normalize: preprocessForm.normalize,
                    encode_categorical: preprocessForm.encodeCategorical,
                },
            });
            loading.preprocess = false;
            if (r.code !== 0) { ElementPlus.ElMessage.error(r.msg || '预处理失败'); return; }
            ElementPlus.ElMessage.success('预处理任务已创建');
            // 轮询结果
            const pid = r.data.preprocess_id;
            const check = async () => {
                const pr = await GET(`/api/preprocess/${pid}/`);
                if (pr.code === 0) {
                    if (pr.data.status === 'success') {
                        preprocessResult.value = pr.data.summary || {};
                        ElementPlus.ElMessage.success('预处理完成');
                    } else if (pr.data.status === 'failed') {
                        ElementPlus.ElMessage.error('预处理失败: ' + (pr.data.error_message || '未知错误'));
                    } else {
                        setTimeout(check, 1500);
                    }
                }
            };
            setTimeout(check, 1000);
        }

        // ── 防御/清洗 ─────────────────────────────────────────────
        function goToDefense() {
            if (!currentTask.value || currentTask.value.status !== 'success') {
                ElementPlus.ElMessage.warning('请先完成检测任务'); return;
            }
            defenseStrategy.default = 'remove';
            defenseStrategy.relabel = {};
            defenseStrategy.ignore = [];
            for (const r of (currentTask.value.results || [])) {
                if (r.suggestion === 'relabel') defenseStrategy.relabel[r.sample_id] = '';
                else if (r.suggestion === 'ignore') defenseStrategy.ignore.push(r.sample_id);
            }
            defenseResults.value = null;
            page.value = 'defense';
        }

        async function applyDefense() {
            if (!currentTask.value) return;
            loading.defense = true;
            const strategy = { default: defenseStrategy.default };
            if (Object.keys(defenseStrategy.relabel).length > 0) strategy.relabel = defenseStrategy.relabel;
            if (defenseStrategy.ignore.length > 0) strategy.ignore = defenseStrategy.ignore;
            const r = await POST(`/api/defense/tasks/${currentTask.value.task_id}/apply/`, { strategy });
            loading.defense = false;
            if (r.code !== 0) { ElementPlus.ElMessage.error(r.msg || '无害化处理失败'); return; }
            defenseResults.value = r.data;
            ElementPlus.ElMessage.success('无害化处理完成');
        }

        async function downloadCleanDataset() {
            if (!defenseResults.value?.clean_result_id) return;
            window.open(`/api/defense/${defenseResults.value.clean_result_id}/download/`, '_blank');
        }

        // ── 报告 ──────────────────────────────────────────────────
        async function generateReport() {
            if (!currentTask.value) return;
            loading.report = true;
            const r = await POST('/api/reports/', { task_id: currentTask.value.task_id, report_type: 'html' });
            loading.report = false;
            if (r.code !== 0) { ElementPlus.ElMessage.error(r.msg || '报告生成失败'); return; }
            reportInfo.value = r.data;
            ElementPlus.ElMessage.success('报告已生成');
        }

        async function generateReportForTask(taskItem) {
            const r = await POST('/api/reports/', { task_id: taskItem.task_id, report_type: 'html' });
            if (r.code !== 0) { ElementPlus.ElMessage.error(r.msg || '报告生成失败'); return; }
            ElementPlus.ElMessage.success('报告已生成');
            window.open(`/api/reports/${r.data.report_id}/download/`, '_blank');
        }

        function downloadReport() {
            if (!reportInfo.value?.report_id) return;
            window.open(`/api/reports/${reportInfo.value.report_id}/download/`, '_blank');
        }

        // ── 辅助函数 ──────────────────────────────────────────────
        function statusType(s) { return { queued:'info', running:'warning', success:'success', failed:'danger', cancelled:'' }[s] || ''; }
        function statusLabel(s) { return { queued:'排队中', running:'执行中', success:'已完成', failed:'失败', cancelled:'已取消' }[s] || s; }
        function riskColor(score) {
            if (score == null) return 'var(--text-muted)';
            if (score >= 0.15) return 'var(--danger)';
            if (score >= 0.05) return 'var(--warning)';
            return 'var(--success)';
        }
        function riskLabel(score) {
            if (score == null) return '未知';
            if (score >= 0.15) return '高风险';
            if (score >= 0.05) return '中风险';
            return '低风险';
        }
        function riskBg(score) {
            if (score == null) return 'rgba(100,116,139,0.15)';
            if (score >= 0.15) return 'rgba(239,68,68,0.15)';
            if (score >= 0.05) return 'rgba(245,158,11,0.15)';
            return 'rgba(16,185,129,0.15)';
        }
        function riskConclClass(score) {
            if (score == null) return 'conclusion-low';
            if (score >= 0.15) return 'conclusion-high';
            if (score >= 0.05) return 'conclusion-medium';
            return 'conclusion-low';
        }
        function riskGradient(score) {
            if (score >= 0.15) return 'linear-gradient(90deg, #ef4444, #dc2626)';
            if (score >= 0.05) return 'linear-gradient(90deg, #f59e0b, #d97706)';
            return 'linear-gradient(90deg, #10b981, #059669)';
        }
        function riskBarWidth(n) { return Math.min(n * 20, 100); }
        function riskTypeColor(t) {
            return { label_poison:'var(--danger)', backdoor:'var(--purple)',
                distribution_shift:'var(--accent)', anomaly:'var(--warning)', unknown:'var(--text-muted)' }[t] || 'var(--text-muted)';
        }
        function confBucketColor(i) { return ['var(--text-muted)', 'var(--text-secondary)', 'var(--warning)', '#f09035', 'var(--danger)'][i]; }
        function sugStyle(s) {
            const map = {
                remove:  { bg: 'rgba(239,68,68,0.08)', border: 'rgba(239,68,68,0.3)', text: 'var(--danger)' },
                relabel: { bg: 'rgba(245,158,11,0.08)', border: 'rgba(245,158,11,0.3)', text: 'var(--warning)' },
                review:  { bg: 'rgba(59,130,246,0.08)', border: 'rgba(59,130,246,0.3)', text: 'var(--accent)' },
                ignore:  { bg: 'rgba(16,185,129,0.08)', border: 'rgba(16,185,129,0.3)', text: 'var(--success)' },
            };
            return map[s] || { bg: 'var(--bg-input)', border: 'var(--border)', text: 'var(--text-secondary)' };
        }
        function detectorTypeLabel(t) { return { label_poison:'标签投毒', anomaly:'异常检测', distribution:'分布漂移', backdoor:'后门检测', influence:'影响函数' }[t] || t; }
        function detectorTypeTag(t) { return { label_poison:'danger', anomaly:'warning', distribution:'', backdoor:'info', influence:'success' }[t] || ''; }
        function riskTypeLabel(t) { return { label_poison:'标签投毒', backdoor:'后门攻击', distribution_shift:'分布偏移', anomaly:'异常样本', unknown:'未知' }[t] || t; }
        function suggestionLabel(s) { return { remove:'建议删除', relabel:'建议重标', ignore:'忽略', review:'人工复查' }[s] || s; }
        function formatSize(bytes) {
            if (!bytes) return '-';
            if (bytes > 1024 * 1024) return (bytes / 1024 / 1024).toFixed(1) + ' MB';
            return (bytes / 1024).toFixed(1) + ' KB';
        }
        function formatDate(d) {
            if (!d) return '-';
            return new Date(d).toLocaleString('zh-CN', { hour12: false });
        }
        function getDatasetName(id) {
            const ds = datasets.value.find(d => d.id === id);
            return ds ? ds.dataset_name : (id || '-');
        }
        function dtypeTag(dtype) {
            if (!dtype) return 'info';
            if (dtype.includes('int') || dtype.includes('float')) return 'warning';
            if (dtype.includes('bool')) return 'success';
            return 'info';
        }
        function getDetectorDisplayName(name) {
            const d = detectors.value.find(det => det.detector_name === name);
            return d ? d.display_name : name;
        }

        // ── 页面切换监听 ──────────────────────────────────────────
        onMounted(() => { if (isLoggedIn.value) loadAll(); });
        watch(page, (np) => {
            if (np === 'tasks') { loadTasks(); startPolling(); }
            else if (np === 'dashboard') { loadDashboard(); loadTasks(); }
            else if (np === 'admin') { loadAdminDashboard(); }
            else if (np === 'users') { loadAdminUsers(); }
            else if (np === 'logs') { loadAuditLogs(); }
            else if (np === 'datasets') { loadDatasets(); }
            else if (np === 'methods') { loadMethods(); }
            else stopPolling();
        });

        return {
            token, user, page, loading, loginForm, loginErr,
            datasets, tasks, methods, detectors, currentTask,
            step, uploadFile, uploadedId, submitForm,
            datasetDetail, showDatasetDetail, showUploadDialog,
            dsUploadFile, dsUploadForm,
            datasetPreviewRows, datasetPreviewCols, datasetPreviewTotal,
            preprocessForm, preprocessResult,
            defenseResults, defenseStrategy, reportInfo, analysisData,
            dashStats, adminStats, adminUsers,
            auditLogs, auditLogTotal, auditLogPage, auditPageSize, logFilter,
            isLoggedIn, completedTasks, pageTitle,
            datasetColumns, reportStats, currentDataset, conclusionText,
            selectedRequiresBaseline,
            login, logout, loadDatasets, loadTasks, loadMethods, loadDetectors,
            loadAdminUsers, loadAuditLogs, toggleUser,
            handleFileChange, doUpload, useExistingDataset,
            openUploadDialog, handleDsUploadChange, doDsUpload,
            openDatasetDetail, startDetectFromDs, deleteDataset,
            toggleDetector, submitTask, viewTask, closeTask,
            runPreprocess, goToDefense, applyDefense, downloadCleanDataset,
            generateReport, generateReportForTask, downloadReport,
            statusType, statusLabel, riskColor, riskLabel, riskBg,
            riskConclClass, riskGradient, riskBarWidth, riskTypeColor, riskTypeLabel,
            confBucketColor, sugStyle, detectorTypeLabel, detectorTypeTag,
            suggestionLabel, formatSize, formatDate, getDatasetName, dtypeTag,
            getDetectorDisplayName,
        };
    },
});

for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
    app.component(key, component);
}
app.use(ElementPlus);
app.mount('#app');
