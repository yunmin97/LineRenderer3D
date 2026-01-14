using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;
using Unity.Burst;
[System.Serializable]
public class LineRenderer3D : MonoBehaviour
{
    public bool autoUpdate;
    public int resolution;
    public Material material;
    MeshFilter meshFilter;
    Mesh mesh;
    MeshRenderer meshRenderer;
    [SerializeField] List<Point> points = new List<Point>();
    bool autoComplete;
    //-----------------------------------------------------------------------//
    NativeArray<Vector3> vertices;
    NativeArray<Vector3> normals;
    NativeArray<Vector2> uvs;
    NativeArray<Point> nodes;
    NativeArray<int> indices;
    NativeArray<float> sines;
    NativeArray<float> cosines;
    JobHandle jobHandle;
    JobHandle pointsJobHandle;
    JobHandle rotationJobHandle;
    NativeArray<float> distances; // 每个点的累计长度
    void Awake(){
        meshRenderer = gameObject.AddComponent<MeshRenderer>();
        meshRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        meshRenderer.reflectionProbeUsage = UnityEngine.Rendering.ReflectionProbeUsage.BlendProbes;
        meshFilter = gameObject.AddComponent<MeshFilter>();
        meshFilter.sharedMesh = new Mesh();
    }
    void Start()
    {
        mesh = new Mesh();
        meshFilter.sharedMesh = mesh;
        meshRenderer.sharedMaterial = material;
    }
    void Update()
    {
        if (autoUpdate && points.Count > 1)
            BeginGeneration();
    }
    void LateUpdate(){
        if (points.Count <= 1)
            return;

        if (autoUpdate) 
            CompleteGeneration();
        else if(autoComplete){
            CompleteGeneration();
            autoComplete = false;
        }
    }
    void OnDestroy()
    {
        CompleteAndDisposeJobs();
    }

    void CompleteAndDisposeJobs()
    {
        jobHandle.Complete();
        pointsJobHandle.Complete();
        rotationJobHandle.Complete();

        DisposeIfCreated(ref vertices);
        DisposeIfCreated(ref indices);
        DisposeIfCreated(ref nodes);
        DisposeIfCreated(ref normals);
        DisposeIfCreated(ref uvs);
        DisposeIfCreated(ref sines);
        DisposeIfCreated(ref cosines);
        DisposeIfCreated(ref distances);
    }
    static void DisposeIfCreated<T>(ref NativeArray<T> arr) where T : struct
    {
        if (arr.IsCreated)
        {
            arr.Dispose();
            arr = default;
        }
    }
    public void BeginGenerationAutoComplete(){
        BeginGeneration();
        autoComplete = true;
    }
    public void BeginGeneration(){
        CompleteAndDisposeJobs();

        if (points == null || points.Count <= 1)
        {
            // 确保不显示残留网格
            if (mesh != null)
                mesh.Clear();
            return;
        }

        int ringVertexCount = points.Count * resolution;
        int capVertexCount = 2; // 头 + 尾

        vertices = new NativeArray<Vector3>(ringVertexCount + capVertexCount, Allocator.Persistent);
        normals = new NativeArray<Vector3>(ringVertexCount + capVertexCount, Allocator.Persistent);
        uvs = new NativeArray<Vector2>(ringVertexCount + capVertexCount, Allocator.Persistent);

        // 原有管子三角 + 头尾端盖三角
        int tubeIndexCount = (points.Count - 1) * resolution * 6;
        int capIndexCount = resolution * 3 * 2;

        indices = new NativeArray<int>(tubeIndexCount + capIndexCount, Allocator.Persistent);

        nodes = new NativeArray<Point>(points.Count, Allocator.Persistent);
        sines = new NativeArray<float>(resolution, Allocator.Persistent);
        cosines = new NativeArray<float>(resolution, Allocator.Persistent);
        for(int i = 0; i < points.Count; i++){
            nodes[i] = points[i];
        }

        var pointsJob = new CalculatePointData()
        {
            nodes = nodes
        };
        pointsJobHandle = pointsJob.Schedule(points.Count - 1, 32);
        for(int i = 0; i < resolution; i++){
            sines[i] = Mathf.Sin(i * Mathf.PI * 2 / resolution);
            cosines[i] = Mathf.Cos(i * Mathf.PI * 2 / resolution);
        }
        pointsJobHandle.Complete();
        CalculateEdgePoints(); 

        var rotationJob = new FixPointsRotation()
        {
            nodes = nodes
        };
        rotationJobHandle = rotationJob.Schedule();
        rotationJobHandle.Complete(); //uses job only to utilize burst system for better performance

        distances = new NativeArray<float>(points.Count, Allocator.Persistent);
        distances[0] = 0f;
        for (int i = 1; i < points.Count; i++)
        {
            distances[i] =
                distances[i - 1] +
                Vector3.Distance(nodes[i].position, nodes[i - 1].position);
        }

        var meshJob = new Line3D() {
            resolution = resolution,
            indices = indices,
            vertices = vertices,
            sines = sines,
            nodes = nodes,
            cosines = cosines,
            normals = normals,
            uvs = uvs,
            iterations = points.Count,
            distances = distances,   // 新增
        };
        jobHandle = meshJob.Schedule(points.Count, 16);
        JobHandle.ScheduleBatchedJobs();
    }
    public void CompleteGeneration(){
        if (!jobHandle.IsCompleted || points.Count <= 1)
            return;

        jobHandle.Complete();

        int ringVertexCount = points.Count * resolution;
        int headCenterIndex = ringVertexCount;
        int tailCenterIndex = ringVertexCount + 1;

        // ===== 头部中心点 =====
        vertices[headCenterIndex] = nodes[0].position;
        normals[headCenterIndex] = -nodes[0].direction.normalized;
        uvs[headCenterIndex] = new Vector2(0, 0.5f);

        // ===== 尾部中心点 =====
        vertices[tailCenterIndex] = nodes[nodes.Length - 1].position;
        normals[tailCenterIndex] = nodes[nodes.Length - 1].direction.normalized;
        uvs[tailCenterIndex] = new Vector2(1, 0.5f);

        int indexOffset = (points.Count - 1) * resolution * 6;

        // ===== 头部端盖（反向，朝外）=====
        for (int j = 0; j < resolution; j++)
        {
            int next = (j + 1) % resolution;

            indices[indexOffset++] = headCenterIndex;
            indices[indexOffset++] = next;
            indices[indexOffset++] = j;
        }

        // ===== 尾部端盖（正向）=====
        int baseRing = (points.Count - 1) * resolution;

        for (int j = 0; j < resolution; j++)
        {
            int next = (j + 1) % resolution;

            indices[indexOffset++] = tailCenterIndex;
            indices[indexOffset++] = baseRing + j;
            indices[indexOffset++] = baseRing + next;
        }

        mesh.Clear();
        mesh.SetVertices(vertices);
        mesh.SetIndices(indices, MeshTopology.Triangles, 0);
        mesh.SetNormals(normals);
        mesh.SetUVs(0, uvs);
        mesh.RecalculateBounds();//未设置 Bounds → 视锥裁剪错误,管子可能“消失”

        vertices.Dispose();
        indices.Dispose();
        sines.Dispose();
        cosines.Dispose();
        nodes.Dispose();
        normals.Dispose();
        uvs.Dispose();
    }
    void CalculateEdgePoints(){
        Vector3 edgeRight, edgeUp;
        Vector3 edgeDirection = (nodes[1].position - nodes[0].position).normalized;
        ComputeBasis(edgeDirection, out edgeRight, out edgeUp);
        nodes[0] = new Point(nodes[0].position, edgeDirection, Vector3.zero, edgeUp, edgeRight, nodes[0].thickness);
        edgeDirection = (nodes[nodes.Length-1].position - nodes[nodes.Length-2].position).normalized;
        ComputeBasis(edgeDirection, out edgeRight, out edgeUp);
        nodes[nodes.Length-1] = new Point(nodes[nodes.Length-1].position, edgeDirection, Vector3.zero, edgeUp, edgeRight, nodes[nodes.Length-1].thickness); 
    }

    /// <summary>Unified secure computing right/up method </summary>
    static void ComputeBasis(Vector3 direction, out Vector3 right, out Vector3 up)
    {
        Vector3 reference = Mathf.Abs(Vector3.Dot(direction, Vector3.up)) < 0.99f ? Vector3.up : Vector3.forward;
        right = Vector3.Cross(reference, direction).normalized;
        up = Vector3.Cross(direction, right).normalized;
    }
    ///<summary> initialize renderer with set amount of empty points </summary>
    public void SetPositions(int positionCount){
        points.Clear();
        Point p = new Point(Vector3.zero, 0);
        for(int i = 0; i < positionCount; i++){
            points.Add(p);
        }
    }
    ///<summary> remove point at index </summary>
    public void RemovePoint(int index){
        points.RemoveAt(index);
    }
    ///<summary> add new point </summary>
    public void AddPoint(Vector3 position, float thickness){
        points.Add(new Point(position, thickness));
    }
    ///<summary> change point at index </summary>
    public void SetPoint(int index, Vector3 position, float thickness){
        points[index] = new Point(position, thickness);
    }
    ///<summary> set points to an array of vector3 with uniform thickness </summary>
    public void SetPoints(Vector3[] positions, float thickness){
        points = positions.Select(position => new Point(position, thickness)).ToList();
    }
    public void SetPoints(List<Vector3> positions, float thickness)
    {
        points = positions.Select(position => new Point(position, thickness)).ToList();
    }
    public void SetPoints(List<Point> positions)
    {
        points = positions;
    }

    ///<summary> update material of the mesh </summary>
    public void SetMaterial(Material material)
    {
        this.material = material;
        this.meshRenderer.sharedMaterial = material;
    }

    ///<summary> set points to an array of vector3 and float (thickness) </summary>
    public void SetPoints(Vector3[] positions, float[] thicknesses){
        points = positions.Zip(thicknesses, (position, thickness) => new Point(position, thickness)).ToList();
    }
    ///<summary> get current point count </summary>
    public int Count(){
        return points.Count;
    }
    [System.Serializable]
    public struct Point{
        public Vector3 position;
        [HideInInspector] public Vector3 direction;
        [HideInInspector] public Vector3 normal;
        [HideInInspector] public Vector3 up;
        [HideInInspector] public Vector3 right;
        public float thickness;
        public Point(Vector3 position, Vector3 direction, Vector3 normal, Vector3 up, Vector3 right, float thickness){
            this.position = position;
            this.direction = direction;
            this.normal = normal;
            this.thickness = thickness;
            this.up = up;
            this.right = right;
        }
        public Point(Vector3 position, float thickness){
            this.position = position;
            this.direction = Vector3.zero;
            this.normal = Vector3.zero;
            this.thickness = thickness;
            this.up = Vector3.zero;
            this.right = Vector3.zero;
        }
    }
    [BurstCompile] public struct Line3D : IJobParallelFor {
        public int resolution;
        public int iterations;
        [ReadOnly] public NativeArray<Point> nodes;
        [ReadOnly] public NativeArray<float> sines;
        [ReadOnly] public NativeArray<float> cosines;
        [ReadOnly] public NativeArray<float> distances; // 只读
        //[NativeDisableParallelForRestriction] is unsafe and can cause race conditions,
        //but in this case each job works on n=resolution vertices so it's not an issue
        //look at it like at a 2d array of size Points x resolution
        //i used this approach because it makes adressing points way easier
        [NativeDisableParallelForRestriction] public NativeArray<Vector3> vertices;
        [NativeDisableParallelForRestriction] public NativeArray<int> indices;
        [NativeDisableParallelForRestriction] public NativeArray<Vector3> normals;
        [NativeDisableParallelForRestriction] public NativeArray<Vector2> uvs;
        public void Execute(int i) {
            //双重验证，防止两端变成点
            Vector3 right = nodes[i].right;
            Vector3 up = nodes[i].up;
            if (right.sqrMagnitude < 1e-6f || up.sqrMagnitude < 1e-6f)
                ComputeBasis(nodes[i].direction.normalized, out right, out up);
            right *= nodes[i].thickness;
            up *= nodes[i].thickness;
            //Vector3 right = nodes[i].right.normalized * nodes[i].thickness;
            //Vector3 up = nodes[i].up.normalized * nodes[i].thickness;
            float distance = distances[i]; // 安全
            for (int j = 0; j < resolution; j++){
                vertices[i * resolution + j] = nodes[i].position;
                Vector3 vertexOffset = cosines[j] * right + sines[j] * up;
                normals[i * resolution + j] += vertexOffset.normalized;
                vertexOffset += nodes[i].normal.normalized * Vector3.Dot(nodes[i].normal.normalized, vertexOffset) * (Mathf.Clamp(1/nodes[i].normal.magnitude, 0, 2) - 1);
                vertices[i * resolution + j] += vertexOffset;
                uvs[i * resolution + j] = new(distance, (float)j / (resolution - 1));
                if (i == iterations - 1) continue;
                int offset = i * resolution * 6 + j * 6;
                indices[offset] = j + i * resolution;
                indices[offset + 1] = (j + 1) % resolution + i * resolution;
                indices[offset + 2] = j + resolution + i * resolution;
                indices[offset + 3] = (j + 1) % resolution + i * resolution;
                indices[offset + 4] = (j + 1) % resolution + resolution + i * resolution;
                indices[offset + 5] = j + resolution + i * resolution;
            }
        }
    }
    [BurstCompile] public struct CalculatePointData : IJobParallelFor{
        [NativeDisableParallelForRestriction] public NativeArray<Point> nodes;
        public void Execute(int i){
            if (i <= 0 || i >= nodes.Length - 1) return;
            Vector3 previous = (nodes[i].position - nodes[i-1].position).normalized;
            Vector3 next = (nodes[i+1].position - nodes[i].position).normalized;
            Vector3 direction = Vector3.Lerp(previous, next, 0.5f).normalized;
            Vector3 normal = (next - previous).normalized * Mathf.Abs(Vector3.Dot(previous, direction)); //length encodes cosine of angle   
            ComputeBasis(direction, out Vector3 right, out Vector3 up);
            nodes[i] = new Point(nodes[i].position, direction, normal, up, right, nodes[i].thickness);
        }
    }
    [BurstCompile] public struct FixPointsRotation : IJob{
        public NativeArray<Point> nodes;
        public void Execute(){
                for(int i = 0; i < nodes.Length - 1; i++){
                Vector3 fromTo = (nodes[i+1].position - nodes[i].position).normalized;
                Vector3 firstRight = nodes[i].right - Vector3.Dot(nodes[i].right, fromTo) * fromTo;
                Vector3 secondRight = nodes[i+1].right - Vector3.Dot(nodes[i+1].right, fromTo) * fromTo;
                float angle = -Vector3.SignedAngle(firstRight, secondRight, fromTo);
                Quaternion rot = Quaternion.AngleAxis(angle, nodes[i+1].direction);
                nodes[i+1] = new Point(nodes[i+1].position, nodes[i+1].direction, nodes[i+1].normal, rot * nodes[i+1].up, rot * nodes[i+1].right, nodes[i+1].thickness);
            }   
        }
    }
}
