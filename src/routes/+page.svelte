
<script>
  import { onMount } from 'svelte';
  import pythonScript from '#lib/calc.py?raw';

  // 초기 학습 데이터
  let dataRows = $state([
    { id: 1, Budget: 10, CVR: 15, Users: 150 },
    { id: 2, Budget: 50, CVR: 14, Users: 650 },
    { id: 3, Budget: 100, CVR: 12, Users: 1100 },
    { id: 4, Budget: 200, CVR: 10, Users: 1700 },
    { id: 5, Budget: 400, CVR: 7, Users: 2400 }
  ]);

  let nextId = 6;

  // 상태 관리
  let isLoading = $state(true);
  let isAnalyzing = $state(false);
  let statusMessage = $state('Python 환경 초기화 중...');
  let errorMessage = $state('');

  let linearResult = $state(null);
  let polyResult = $state(null);
  let sampleCount = $state(0);

  let pyodideInstance;

  // 예측용 입력값
  let targetBudget = $state(500000);
  let targetCVR = $state(14);

  // 데이터 변경 감지
  let currentSnapshot = $derived(
    JSON.stringify(
      dataRows.map(({ Budget, CVR, Users }) => ({
        Budget,
        CVR,
        Users
      }))
    )
  );

  let trainingSnapshot = $state('');

  let isDirty = $derived(
    currentSnapshot !== trainingSnapshot
  );

  // Python 초기화
  onMount(() => {
    let cancelled = false;

    async function initializePython() {
      try {
        statusMessage = 'Pyodide 모듈 불러오는 중...';

        const { loadPyodide, version } =
          await import('pyodide');

        if (cancelled) return;

        statusMessage = 'Python 런타임 초기화 중...';

        pyodideInstance = await loadPyodide({
          indexURL:
            `https://cdn.jsdelivr.net/pyodide/v${version}/full/`
        });

        if (cancelled) return;

        statusMessage = 'Python 패키지 설치 중...';

        await pyodideInstance.loadPackage([
          'numpy',
          'pandas',
          'scikit-learn'
        ]);

        if (cancelled) return;

        statusMessage = 'Python 분석 함수 준비 중...';

        await pyodideInstance.runPythonAsync(
          pythonScript
        );

        if (cancelled) return;

        statusMessage = '초기 데이터 분석 중...';

        await trainModel();

      } catch (error) {
        console.error('Pyodide initialization error:', error);

        if (!cancelled) {
          errorMessage = error.message;
        }

      } finally {
        if (!cancelled) {
          isLoading = false;
        }
      }
    }

    initializePython();

    return () => {
      cancelled = true;
    };
  });

  // Python 회귀 모델 학습
  async function trainModel() {
    if (!pyodideInstance) {
      errorMessage = 'Python 실행 환경이 준비되지 않았습니다.';
      return;
    }

    if (dataRows.length < 5) {
      errorMessage = '최소 5개의 데이터가 필요합니다.';
      return;
    }

    isAnalyzing = true;
    errorMessage = '';

    try {
      const inputData = dataRows.map(row => ({
        Budget: Number(row.Budget),
        CVR: Number(row.CVR),
        Users: Number(row.Users)
      }));

      pyodideInstance.globals.set(
        'input_data_json',
        JSON.stringify(inputData)
      );

      const pyResult =
        await pyodideInstance.runPythonAsync(
          'fit_models(input_data_json)'
        );

      try {
        const result = pyResult.toJs({
          dict_converter: Object.fromEntries
        });

        if (!result.linear || !result.poly) {
          throw new Error(
            '회귀 분석 결과가 올바르지 않습니다.'
          );
        }

        const values = [
          result.linear.w_budget,
          result.linear.w_cvr,
          result.linear.intercept,
          result.poly.w1,
          result.poly.w2,
          result.poly.w3,
          result.poly.intercept,
          result.linear.r2,
          result.poly.r2
        ];

        if (!values.every(Number.isFinite)) {
          throw new Error(
            '회귀 계수에 유효하지 않은 값이 있습니다.'
          );
        }

        linearResult = result.linear;
        polyResult = result.poly;
        sampleCount = result.sample_count;

        trainingSnapshot = currentSnapshot;
        statusMessage = '회귀 분석 완료';

      } finally {
        pyResult.destroy();
      }

    } catch (error) {
      console.error('Regression error:', error);
      errorMessage = error.message;

    } finally {
      isAnalyzing = false;
    }
  }

  // 데이터 추가
  function addRow() {
    dataRows.push({
      id: nextId++,
      Budget: 50,
      CVR: 14,
      Users: 500
    });
  }

  // 데이터 삭제
  function removeRow(id) {
    if (dataRows.length <= 5) return;

    dataRows = dataRows.filter(
      row => row.id !== id
    );
  }

  // 예측 입력값 변환
  let budgetInManwon = $derived(
    Number(targetBudget) / 10000
  );

  let cvrValue = $derived(
    Number(targetCVR) / 100
  );

  // 선형 회귀 예측
  let predictedLinearUsers = $derived(
    linearResult
      ? linearResult.w_budget * budgetInManwon
        + linearResult.w_cvr * cvrValue
        + linearResult.intercept
      : 0
  );

  // 2차 다항 회귀 예측
  let predictedPolyUsers = $derived(
    polyResult
      ? polyResult.w1 * budgetInManwon
        + polyResult.w2 * budgetInManwon ** 2
        + polyResult.w3 * cvrValue
        + polyResult.intercept
      : 0
  );

  // 예산에 대한 선형 회귀 편미분
  let linearSlope = $derived(
    linearResult
      ? linearResult.w_budget
      : 0
  );

  // 예산에 대한 다항 회귀 편미분
  let currentSlope = $derived(
    polyResult
      ? polyResult.w1
        + 2 * polyResult.w2 * budgetInManwon
      : 0
  );

  // 예산에 대한 2차 편미분
  let secondDerivative = $derived(
    polyResult
      ? 2 * polyResult.w2
      : 0
  );

  // CVR에 대한 선형 회귀 편미분
  let cvrSlope = $derived(
    linearResult
      ? linearResult.w_cvr
      : 0
  );

  // CVR에 대한 다항 회귀 편미분
  let polyCvrSlope = $derived(
    polyResult
      ? polyResult.w3
      : 0
  );

  // 표시용 숫자
  let formattedBudget = $derived(
    Number(targetBudget).toLocaleString('ko-KR')
  );

  let formattedLinearUsers = $derived(
    Math.round(predictedLinearUsers)
      .toLocaleString('ko-KR')
  );

  let formattedPolyUsers = $derived(
    Math.round(predictedPolyUsers)
      .toLocaleString('ko-KR')
  );

</script>

<svelte:head>
  <title>Smoke Test | 수요 분석</title>
  <meta
    name="description"
    content="Python 회귀 분석과 미분을 이용한 수요 예측"
  />
</svelte:head>

<div class="min-h-screen bg-base-200">
  <main class="mx-auto max-w-6xl space-y-6 px-4 py-8 sm:px-6">

    <!-- Header -->
    <header class="flex flex-wrap items-center justify-between gap-4">
      <div>
        <div class="badge badge-primary badge-outline mb-3">
          PYTHON · REGRESSION · CALCULUS
        </div>

        <h1 class="text-3xl font-bold">
          Smoke Test 수요 분석
        </h1>

        <p class="mt-2 text-sm text-base-content/60">
          광고 예산과 전환율에 따른 가입자 수 예측
        </p>
      </div>

      {#if isLoading || isAnalyzing}
        <span class="badge badge-warning badge-outline">
          <span class="loading loading-spinner loading-xs"></span>
          분석 중
        </span>
      {:else if errorMessage}
        <span class="badge badge-error badge-outline">
          분석 오류
        </span>
      {:else}
        <span class="badge badge-success badge-outline">
          분석 완료
        </span>
      {/if}
    </header>

    {#if isLoading}

      <div class="card bg-base-100 shadow-sm">
        <div class="card-body items-center py-12">
          <span class="loading loading-ring loading-lg text-primary"></span>
          <p class="font-medium">{statusMessage}</p>
        </div>
      </div>

    {:else}

      {#if errorMessage}
        <div role="alert" class="alert alert-error">
          <span>{errorMessage}</span>
        </div>
      {/if}

      <!-- Dataset Editor -->
      <section class="card border border-base-300 bg-base-100 shadow-sm">
        <div class="card-body">

          <div class="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 class="card-title">
                학습 데이터 편집
              </h2>

              <p class="text-sm text-base-content/60">
                Budget, CVR, Users 값을 직접 수정할 수 있습니다.
              </p>
            </div>

            <span class="badge badge-neutral">
              {dataRows.length}개 데이터
            </span>
          </div>

          <div class="overflow-x-auto">
            <table class="table table-zebra table-sm">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Budget (만 원)</th>
                  <th>CVR (%)</th>
                  <th>Users (명)</th>
                  <th></th>
                </tr>
              </thead>

              <tbody>
                {#each dataRows as row, i (row.id)}
                  <tr>
                    <th>{i + 1}</th>

                    <td>
                      <input
                        type="number"
                        min="0.01"
                        step="1"
                        bind:value={row.Budget}
                        class="input input-bordered input-sm w-28"
                        aria-label={`데이터 ${i + 1} Budget`}
                      />
                    </td>

                    <td>
                      <input
                        type="number"
                        min="0"
                        max="100"
                        step="0.1"
                        bind:value={row.CVR}
                        class="input input-bordered input-sm w-24"
                        aria-label={`데이터 ${i + 1} CVR`}
                      />
                    </td>

                    <td>
                      <input
                        type="number"
                        min="0"
                        step="1"
                        bind:value={row.Users}
                        class="input input-bordered input-sm w-28"
                        aria-label={`데이터 ${i + 1} Users`}
                      />
                    </td>

                    <td>
                      <button
                        class="btn btn-ghost btn-xs text-error"
                        onclick={() => removeRow(row.id)}
                        disabled={dataRows.length <= 5}
                        aria-label={`데이터 ${i + 1} 삭제`}
                      >
                        삭제
                      </button>
                    </td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>

          <div class="flex flex-wrap justify-between gap-3">
            <button
              class="btn btn-outline btn-sm"
              onclick={addRow}
            >
              + 데이터 추가
            </button>

            <button
              class="btn btn-primary"
              onclick={trainModel}
              disabled={isAnalyzing || dataRows.length < 5}
            >
              {#if isAnalyzing}
                <span class="loading loading-spinner loading-sm"></span>
                재학습 중...
              {:else}
                회귀 분석 실행
              {/if}
            </button>
          </div>

          {#if isDirty}
            <div class="alert alert-warning alert-soft text-sm">
              데이터가 변경되었습니다. 회귀 분석 실행을 눌러 모델을 다시 학습하세요.
            </div>
          {:else if linearResult}
            <div class="text-xs text-success">
              현재 입력 데이터로 학습된 모델입니다.
            </div>
          {/if}

        </div>
      </section>

      <!-- Prediction Controls -->
      <section class="card border border-base-300 bg-base-100 shadow-sm">
        <div class="card-body gap-6">

          <div>
            <h2 class="card-title">
              예측 조건 설정
            </h2>

            <p class="text-sm text-base-content/60">
              학습된 모델에 새로운 Budget과 CVR을 입력합니다.
            </p>
          </div>

          <!-- Budget -->
          <div class="rounded-box bg-base-200 p-5">
            <div class="flex items-center justify-between">
              <label for="budget" class="font-semibold">
                광고 예산
              </label>

              <span class="text-xl font-bold text-primary">
                {formattedBudget}원
              </span>
            </div>

            <input
              id="budget"
              type="range"
              min="100000"
              max="5000000"
              step="100000"
              bind:value={targetBudget}
              class="range range-primary mt-4 w-full"
            />

            <div class="flex justify-between text-xs text-base-content/50">
              <span>10만 원</span>
              <span>250만 원</span>
              <span>500만 원</span>
            </div>
          </div>

          <!-- CVR -->
          <div class="rounded-box bg-base-200 p-5">
            <div class="flex items-center justify-between">
              <label for="cvr" class="font-semibold">
                전환율 (CVR)
              </label>

              <span class="text-xl font-bold text-success">
                {targetCVR}%
              </span>
            </div>

            <input
              id="cvr"
              type="range"
              min="1"
              max="30"
              step="1"
              bind:value={targetCVR}
              class="range range-success mt-4 w-full"
            />

            <div class="flex justify-between text-xs text-base-content/50">
              <span>1%</span>
              <span>15%</span>
              <span>30%</span>
            </div>
          </div>

        </div>
      </section>

      <!-- Prediction Results -->
      {#if linearResult && polyResult}

        <section>
          <div class="mb-4 flex flex-wrap items-center justify-between gap-3">
            <h2 class="text-xl font-bold">
              예측 결과
            </h2>

            <span class="badge badge-info badge-outline">
              학습 데이터 {sampleCount}개
            </span>
          </div>

          <div class="grid gap-4 md:grid-cols-2">

            <!-- Linear -->
            <div class="card border border-base-300 bg-base-100 shadow-sm">
              <div class="card-body">
                <div class="flex justify-between">
                  <h3 class="card-title">선형 회귀</h3>
                  <span class="badge badge-info">Linear</span>
                </div>

                <div class="stat px-0">
                  <div class="stat-title">예측 가입자 수</div>
                  <div class="stat-value text-info">
                    {formattedLinearUsers}
                  </div>
                  <div class="stat-desc">명</div>
                </div>

                <div class="rounded-box bg-base-200 p-4">
                  <p class="text-xs text-base-content/60">
                    결정계수 R²
                  </p>
                  <p class="text-xl font-bold">
                    {linearResult.r2.toFixed(4)}
                  </p>
                </div>
              </div>
            </div>

            <!-- Polynomial -->
            <div class="card border border-primary/30 bg-base-100 shadow-sm">
              <div class="card-body">
                <div class="flex justify-between">
                  <h3 class="card-title">2차 다항 회귀</h3>
                  <span class="badge badge-primary">Polynomial</span>
                </div>

                <div class="stat px-0">
                  <div class="stat-title">예측 가입자 수</div>
                  <div class="stat-value text-primary">
                    {formattedPolyUsers}
                  </div>
                  <div class="stat-desc">명</div>
                </div>

                <div class="rounded-box bg-base-200 p-4">
                  <p class="text-xs text-base-content/60">
                    결정계수 R²
                  </p>
                  <p class="text-xl font-bold">
                    {polyResult.r2.toFixed(4)}
                  </p>
                </div>
              </div>
            </div>

          </div>
        </section>

        <!-- Calculus -->
        <section class="card border border-base-300 bg-base-100 shadow-sm">
          <div class="card-body">
            <div>
              <div class="badge badge-secondary badge-outline mb-2">
                CALCULUS
              </div>

              <h2 class="card-title">
                미분 분석
              </h2>

              <p class="text-sm text-base-content/60">
                현재 예산과 전환율에서의 변화율입니다.
              </p>
            </div>
            <!-- Linear Budget Derivative -->
            <div class="grid gap-4 md:grid-cols-2">
                <div class="rounded-box bg-base-200 p-5">
                <p class="text-sm text-base-content/60">
                    예산에 대한 선형 모델 편미분
                </p>

                <code class="text-lg font-semibold">
                    ∂f/∂B = wB
                </code>

                <p class="mt-3 text-2xl font-bold tabular-nums">
                    {linearSlope.toFixed(4)}
                </p>

                <p class="text-xs text-base-content/50">
                    예산 1만 원 증가 시 예측 가입자 변화량
                </p>
            </div>
            <!-- Budget first derivative -->
            <div class="rounded-box bg-base-200 p-5">
                <p class="text-sm text-base-content/60">
                  예산에 대한 1차 편미분
                </p>

                <code class="text-lg font-semibold">
                  ∂f/∂B = w₁ + 2w₂B
                </code>

                <p class="mt-3 text-2xl font-bold tabular-nums">
                  {currentSlope.toFixed(4)}
                </p>

                <p class="text-xs text-base-content/50">
                  가입자 수 / 만 원
                </p>
            </div>

              <!-- Budget second derivative -->
            <div class="rounded-box bg-base-200 p-5">
                <p class="text-sm text-base-content/60">
                  예산에 대한 2차 편미분
                </p>

                <code class="text-lg font-semibold">
                  ∂²f/∂B² = 2w₂
                </code>

                <p class="mt-3 text-2xl font-bold tabular-nums">
                  {secondDerivative.toFixed(6)}
                </p>
            </div>

              <!-- CVR linear derivative -->
            <div class="rounded-box bg-base-200 p-5">
                <p class="text-sm text-base-content/60">
                  CVR에 대한 선형 모델 편미분
                </p>

                <code class="text-lg font-semibold">
                  ∂f/∂C = wC
                </code>

                <p class="mt-3 text-2xl font-bold tabular-nums">
                  {cvrSlope.toFixed(4)}
                </p>

                <p class="text-xs text-base-content/50">
                  CVR 1.0 변화 기준
                </p>
            </div>

              <!-- CVR polynomial derivative -->
            <div class="rounded-box bg-base-200 p-5">
                <p class="text-sm text-base-content/60">
                  CVR에 대한 다항 모델 편미분
                </p>

                <code class="text-lg font-semibold">
                  ∂f/∂C = w₃
                </code>

                <p class="mt-3 text-2xl font-bold tabular-nums">
                  {polyCvrSlope.toFixed(4)}
                </p>

                <p class="text-xs text-base-content/50">
                  CVR 1.0 변화 기준
                </p>
            </div>

            </div>

            <div class="alert alert-info alert-soft">
              <div>
                <h3 class="font-bold">미분 해석</h3>

                <p class="text-sm">
                  예산에 대한 편미분은 CVR을 고정한 상태에서
                  예산이 변할 때 예측 가입자 수가 얼마나 변하는지를 나타냅니다.
                </p>

                <p class="text-sm">
                  CVR에 대한 편미분은 예산을 고정한 상태에서
                  전환율이 변할 때 예측 가입자 수의 변화량을 나타냅니다.
                </p>
              </div>
            </div>

          </div>
        </section>

      {/if}
    {/if}
  </main>
</div>