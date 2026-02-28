<template>
  <input type="file" id="fileInput"  accept=".mlir" @change="handleFileChange" style="display: none">
  <button @click="openFilePicker">📁 选择final.mlir</button>
  <div class="container">
        <h1>空间利用率峰值图</h1>
        <div id="chart-container">
            <canvas id="mainChart" width="1500" height = "400"></canvas>
            <div class="legend" id="legend"></div>
        </div>
        <div id="tooltip"></div>
  </div>
  <div class="container-gantt">
        <h1>内存分配甘特图</h1>
        <div class="controls">
            <button class="button" @click="resetView">重置视图</button>
            <button class="button" @click="zoomIn">放大</button>
            <button class="button" @click="zoomOut">缩小</button>
        </div>

        <div id="chart-container-gantt">
            <canvas id="ganttChart" width="1500" height="600"></canvas>
        </div>
        <!-- <div class="slider-container">
            <input type="range" id="timelineSlider" class="timeline-slider" min="0" max="100" value="0">
            <div id="timeLabels" class="time-labels"></div>
        </div> -->

        <!-- Legend items will be added here dynamically -->
        <div class="legend" id="legendContainer">
        </div>
        <div id="tooltip-gantt"></div>
        <div class="info-panel">
            <h3>图表说明：</h3>
            <ul>
                <li>X轴表示时间轴</li>
                <li>Y轴表示内存地址空间</li>
                <li>每个彩色矩形表示一个进程在特定时间段占用的内存区域</li>
                <li>矩形显示tensor名称</li>
            </ul>
        </div>
    </div>

</template>



<script setup>
import { ref } from 'vue';

// 使用 ref 创建响应式数据,全局变量
const utils = ref([]);
const iscv181x = ref(0);
const intervals = ref([]);
const neuron_size = ref(0);
const colors = ref([]);
const colors_gantt = ref([]);
const Segments = ref([]);
const mapping_locs_name = ref({});
const mapping_index_name = ref({});
const subnet_ios = ref([]);// 子网索引:{"input":[子网index_输出index or %*],"output":[%?#?]}
const mapping_tensor_subnetID = ref({});
const gantt_log = ref([]);
const input_nums = ref(0);
const output_nums = ref(0);

//解析行中算子的起始地址、内存空间大小、结束地址、数据类型、location
//解析输入和输出的%index 对应loc的map
function parse_ioinfo(line, neuronAddr, locs, output_stage=false) {
  if (line.includes("return"))  return [];

  // const tensorPattern = /tensor<([^>]+): i64>/g;
  const tensorPattern = /tensor<([\s\S]*?): i64>/g;
  const matches = line.match(tensorPattern);//[...line.matchAll(tensorPattern)];
  const Infos = [];
  const multi_output = (output_stage && matches.length > 1)? 1: 0;

  if (!matches){
    return Infos
  }
  for (let i = 0; i < matches.length; i++) {
    const match = matches[i];
    if (!match.includes(": i64")){
      continue;
    }
    let [shapeDtype, addressStr] = match.split(' ');
    if (match.includes("uniform")){
      shapeDtype = match.split(' ')[0];
      addressStr = match.split(' ').at(-3);
    }
    shapeDtype = shapeDtype.slice(7);
    let shapes = shapeDtype.split('x').slice(0, -1); // 移除数据类型部分
    const dtype = shapeDtype.split('x').pop();
    let dateLen;
    if (dtype.includes('i8') || dtype.includes('u8')) {
      dateLen = 1;
    } else if (dtype.includes('f32') || dtype.includes('i32')) {
      dateLen = 4;
    } else if (dtype.includes('16')) {
      dateLen = 2;
    } else {
      throw new Error(`dtype_error: ${dtype}`);
    }

    let tensorSize = dateLen;
    for (const shape of shapes) {
      tensorSize *= parseInt(shape);
    }
    const start = parseInt(addressStr);
    const end = Math.floor((start + tensorSize + 4095) / 4096) * 4096;

    Infos.push({
      start: start - neuronAddr,
      end: end - neuronAddr,
      shape: shapeDtype,
      locs: multi_output ? mapping_locs_name.value[locs][i] : locs
    });
  }
  return Infos;
}

function parse_fused_locs(mlirContent){
  const patternFused = /#loc\d+/g;
  const mapping = new Map();
  const lines = mlirContent.split('\n');
  for(let i = 0; i < lines.length; i++){
    const line = lines[i];
    if (line.includes("loc(fused[#loc")){
      const matches = line.match(patternFused);
      mapping[matches[0]] = matches.slice(1);
      continue;
    }
    else if (line.includes("= loc(")) {
      mapping[line.split(' ')[0]]=line.split('"')[1];
    }
  }
  return mapping;
}

//解析final_mlir中的算子行
function parseFinalMlir_allsubnet(mlirContent) {
  mapping_locs_name.value = parse_fused_locs(mlirContent);// i.e. key '#loc729' values ['#loc28', '#loc29']
  // 使用正则表达式提取neuron_addr、size、chip
  let neuronAddrMatch = mlirContent.match(/module\.neuron_addr\s*=\s*(\d+)\s*:\s*i64/);
  const neuronSizeMatch = mlirContent.match(/module\.neuron_size\s*=\s*(\d+)\s*:\s*i64/);
  const chip = mlirContent.match(/module\.chip\s*=\s*"([^"]+)"/);
  if (!neuronAddrMatch) {
    neuronAddrMatch = [0,0];
    // throw new Error("module.neuron_addr not found");
  }
  if (!neuronSizeMatch) {
    throw new Error("module.neuron_size not found");
  }
  if (!chip) {
    throw new Error("chip not found");
  }
  iscv181x.value = chip[1] == "cv181x" ? 1 : 0;
  const neuronAddr = parseInt(neuronAddrMatch[1]);
  const neuronSize = parseInt(neuronSizeMatch[1]);

  const ioinfo = [];
  let main_inputs = [];
  let idx = 0;
  let subnet_cnt = 0;
  // 解析main的输入和subnet
  const mainContent = mlirContent.split('return')[0].split("\n")
  for (let i = 0; i < mainContent.length; i++){
    const line = mainContent[i];
    if (line.includes("func.func @main")) {
      const [input_str,output_str] = line.split("->")
      input_nums.value = (input_str.match(/args/g) || []).length;
      output_nums.value = (output_str.match(/tensor/g) || []).length;
    }

    if (line.includes("top.Input")){
      input_nums.value += 1
      let loc = line.split(" ").at(-1).slice(4,-1);
      // let args = line.split('top.Input"(')[1].split(')')[0]
      let input = line.split(" =")[0].trim();
      mapping_index_name.value[input] = {"start_ts": idx, "end_ts": 1e9,'locs':loc,'op':"Input"};
      main_inputs.push(input);
      mapping_tensor_subnetID.value[input] = "main";
      ioinfo.push({
        idx: idx++,
        op: "Input_"+loc,
        input_info: [],
        output_info: parse_ioinfo(line.split("->")[1], neuronAddr, loc)
      });
    }
    if (line.includes("call @subfunc_")){
      let subnet_io = {inputs:[], outputs:[], real_inputs:[], real_outputs:[]}
      let left = line.split("=")[0].trim()
      let right = line.split(")")[0].split("(")[1]
      if (right.includes(", ")){//多输入
        subnet_io.inputs = right.split(", ").map(input => {
          if (input in main_inputs) {
            return input;
          }
          return mapping_tensor_subnetID.value[input]+"_"+input;
        });
      } else {///单输入
        if (right in main_inputs) {
          subnet_io.inputs.push(right);
          }
        subnet_io.inputs.push(mapping_tensor_subnetID.value[right]+"_"+right)
      }
      if (left.includes(":")){//多输出
        let num = Number(left.split(":")[1]);
        for(let j=0; j < num; j++){
          subnet_io.outputs.push(subnet_cnt+"_"+left.split(":")[0]+"#"+j);
          mapping_tensor_subnetID.value[left.split(":")[0]+"#"+j]=subnet_cnt;
        }
      }else{//单输出
        subnet_io.outputs.push(subnet_cnt+"_"+left);
        mapping_tensor_subnetID.value[left]=subnet_cnt;
      }
      subnet_cnt++;
      subnet_ios.value.push(subnet_io)
    }
  }
  // console.log("mapping_tensor_subnetID")
  // console.log(mapping_tensor_subnetID.value)
  // console.log("subnet_ios")
  // console.log(subnet_ios.value)
  for (let subnet_i = 0; subnet_i < subnet_cnt; subnet_i++){
    // 查找subfunc_i函数体
    const regex = new RegExp(`func\\.func @subfunc_${subnet_i}\([^)]*\)[^{]*{`);
    const funcStartMatch = mlirContent.match(regex);
    // const funcStartMatch = mlirContent.match(/func\.func @subfunc_0\([^)]*\)[^{]*{/);
    if (!funcStartMatch) {
      return [];
    }

    let startIndex = funcStartMatch.index + funcStartMatch[0].length;

    // 跳过第一个换行符找到函数体开始
    while (startIndex < mlirContent.length && mlirContent[startIndex] !== "\n") {
      startIndex++;
    }
    startIndex++; // 跳过换行符

    let braceCount = 1;
    let currentIndex = startIndex;
    const n = mlirContent.length;

    // 找到函数体的结束位置
    while (braceCount > 0 && currentIndex < n) {
      if (mlirContent[currentIndex] === '{') {
        braceCount++;
      } else if (mlirContent[currentIndex] === '}') {
        braceCount--;
      }
      currentIndex++;
    }

    const funcBody = mlirContent.substring(startIndex, currentIndex - 1).trim();
    const lines = funcBody.split('\n')
      .map(line => line.trim())
      .filter(line => line.length > 0);

    let inGroup = false;
    let group_line = "";
    // console.log(lines)
    for (const line of lines) {
      if (line.includes("top.Weight") || line.includes("top.None")) {
        continue;
      }

      let loc = line.split(" ").at(-1).slice(4,-1);

      // 提取操作类型
      const opType = line.includes('"') ? line.split('"')[1].slice(4) : "UNKNOWN";
      let opName = subnet_cnt == 1 ? opType + "_" + loc : subnet_i + "_" + opType + "_" + loc;


      // 处理group操作
      if (inGroup) {
        if (line.includes("core_slice_ncdhw")) {
          inGroup = false;
          opName =  subnet_cnt == 1 ?  "Group_" + loc : subnet_i + "_Group_" + loc;
          parse_index_timestep(group_line, opName, idx, subnet_i)
          // 继续处理此行
        } else {
          continue;
        }
      }

      if (line.includes("tpu.Group")) {
        inGroup = true;
        group_line = line;
        continue;
      }

      // 区分输入输出，提取张量信息
      const arrowIndex = line.indexOf('->');
      let inputPart = '';
      let outputPart = '';

      if (arrowIndex !== -1) {
        inputPart = line.substring(0, arrowIndex);
        outputPart = line.substring(arrowIndex + 2);
      } else {
        inputPart = '';
        outputPart = line;
      }

      parse_index_timestep(line, opName, idx, subnet_i);
      ioinfo.push({
        idx: idx++,
        op: opName,
        input_info: parse_ioinfo(inputPart, neuronAddr, loc),
        output_info: parse_ioinfo(outputPart, neuronAddr, loc, true)
      });
    }
  }
  // console.log(ioinfo)
  // console.log(chip.value)
  return {
    neuron_addr: neuronAddr,
    neuron_size: neuronSize,
    info: ioinfo,
  };
}

//解析算子的输入和输出，建立time_step, %27:2 = "tpu.Group"(%24, %20#0) ({...
function parse_index_timestep(line, name_loc, idx, subnet_i=0){

  // return %112, %101 : tensor<1x896x16xf32, 687194767360 : i64>, tensor<1x896x1xf32, 687194857472 : i64> loc(#loc)
  if (line.includes("return")) {
    const right  = line.split(' :')[0].slice(7)
    if (right.includes(",")){//多输入
      for (const part of right.split(", ")) {
        // console.log("in retunr: "+ subnet_i+"_"+part +"  idx: "+idx)
        subnet_ios.value[subnet_i].real_outputs.push(subnet_i+"_"+part);
        mapping_index_name.value[subnet_i+"_"+part]["end_ts"] = Math.max(mapping_index_name.value[subnet_i+"_"+part]["end_ts"], idx);
      }
    } else {//单输入
      // console.log("in retunr: "+ subnet_i+"_"+right +"  idx: "+idx)
      subnet_ios.value[subnet_i].real_outputs.push(subnet_i+"_"+right);
      mapping_index_name.value[subnet_i+"_"+right]["end_ts"] = Math.max(mapping_index_name.value[subnet_i+"_"+right]["end_ts"], idx);
    }
  }

  if (!line.includes("tpu.") && !line.includes("top.")){
    return;
  }

  let opname,loc
  if ((name_loc).split("_").length == 3){
    [subnet_i,opname,loc] = name_loc.split("_");
  } else {
    [opname,loc] = name_loc.split("_");
  }

  const left_right = line.split(') ')[0];
  const left = left_right.split(' = "')[0];
  const right = left_right.split('(')[1];
  if (left.includes(":") || left.includes(",")) {//多输出 可能是类似%8:2或者topk的 %values, %indices
    if (left.includes(":")){ //正常case %8:2
      let input_num = Number(left.split(':')[1]);
      for(let i = 0; i < input_num; i++){
        mapping_index_name.value[subnet_i+"_"+left.split(':')[0]+"#"+i] = {"start_ts": idx, "end_ts": idx, 'locs':mapping_locs_name.value[loc][i], "op":opname};
      }
    } else { //TOPK case
      const parts = left.split(",");
      for (let i = 0; i < parts.length; i++) {
        const part = parts[i].trim();
        mapping_index_name.value[subnet_i+"_"+part] = {"start_ts": idx, "end_ts": idx, 'locs':mapping_locs_name.value[loc][i], "op":opname}
      }
    }

  } else {//单输出
    mapping_index_name.value[subnet_i+"_"+left] = {"start_ts": idx, "end_ts": idx, 'locs':loc, "op":opname};
  }
  // console.log("right "+right)
  // console.log("subnet_ios:")
  // console.log(subnet_ios.value)
  // console.log(mapping_tensor_subnetID.value)
  // console.log("mapping_index_name")
  // console.log(mapping_index_name.value)
  if (right.includes(",")) {//多输入
    for(const part of right.split(", ")){
      if (part.includes("%arg")){
        if (subnet_i==0) continue
        const i_args = Number(part.split("%arg")[1])
        let real_input = subnet_ios.value[Number(subnet_i)].inputs[i_args]
        // console.log("part:  "+part)
        // console.log("i_args: "+i_args)
        // console.log("real_input: "+real_input)
        if (!real_input.includes("main")){
          const from_subnet_i = Number(real_input.split("_")[0]);
          const from_subnet_output_i = real_input.includes("#") ? Number(real_input.split("#")[1]) : 0;
          // console.log("from_subnet_i: "+from_subnet_i)
          // console.log("from_subnet_output_i: "+from_subnet_output_i)
          real_input = subnet_ios.value[from_subnet_i].real_outputs[from_subnet_output_i]
          // console.log("!real_input: "+real_input)
          mapping_index_name.value[real_input]["end_ts"] = Math.max(mapping_index_name.value[real_input]["end_ts"], idx+1);
        }
      }else{
        if (!mapping_index_name.value[subnet_i+"_"+part]) continue
        mapping_index_name.value[subnet_i+"_"+part]["end_ts"] = Math.max(mapping_index_name.value[subnet_i+"_"+part]["end_ts"], idx+1);
      }
    }
  } else {//单输入
    if (right.includes("%arg")){
      const i_args = Number(right.split("%arg")[1])
      const real_input = subnet_ios.value[Number(subnet_i)].inputs[i_args]
      // console.log("i_args: "+i_args)
      // console.log("real_input: "+real_input)
      if (!real_input.includes("main")){
        const from_subnet_i = Number(real_input.split("_")[0]);
        const from_subnet_output_i = real_input.includes("#") ? Number(real_input.split("#")[1]) : 0;
        real_input = subnet_ios.value[from_subnet_i].real_outputs[from_subnet_output_i]
        // console.log("!real_input: "+real_input)
        mapping_index_name.value[real_input]["end_ts"] = Math.max(mapping_index_name.value[real_input]["end_ts"], idx+1);
      }
    } else {
      if (!mapping_index_name.value[subnet_i+"_"+right]) return
      mapping_index_name.value[subnet_i+"_"+right]["end_ts"] = Math.max(mapping_index_name.value[subnet_i+"_"+right]["end_ts"], idx+1);
    }
  }
}

function transform_ts_log(){
  const output = {}
  // %6:{start_ts: 2, end_ts: 4, locs: '#loc8',"op":"name_i"}
  //遍历mapping_index_name所有的key和value，将locs作为key，values整理成这样{'idx': 0, 'op': 'top.Input', 'live_start': 0, 'live_end': 4294967295}
  for (const [key, value] of Object.entries(mapping_index_name.value)) {
    output[mapping_locs_name.value[value.locs]]={
      idx: 999,//key,
      op: value['op'],
      live_start: value['start_ts'],
      live_end: value['end_ts']
    };
  }
  return output;
}

//区间管理类
class IntervalManager {
  constructor(max = 1000000000) {
    this.occupied = []; // 当前占用的区间列表，每个元素是[start, end, opName]数组
    this.free = [];     // 当前空闲的区间列表，每个元素是[start, end]数组
    this.max = max;
  }

  addIntervals(intervals) {
    if (!intervals || intervals.length === 0) {
      return;
    }

    // 将新区间按起始位置排序
    const newIntervals = intervals.sort((a, b) => a[0] - b[0]);

    // 找出与新区间有交集的已占用区间
    const intervalsToRelease = [];
    const remainingOccupied = [];

    for (const occupiedInterval of this.occupied) {
      let shouldRelease = false;
      for (const newInterval of newIntervals) {
        if (this._hasOverlap(occupiedInterval, newInterval)) {
          shouldRelease = true;
          break;
        }
      }

      if (shouldRelease) {
        intervalsToRelease.push(occupiedInterval);
      } else {
        remainingOccupied.push(occupiedInterval);
      }
    }

    // 更新占用区间列表：保留未释放的 + 添加新的
    this.occupied = remainingOccupied.concat(newIntervals);
    this.occupied.sort((a, b) => a[0] - b[0]);
  }

  _hasOverlap(interval1, interval2) {
    const [start1, end1] = interval1;
    const [start2, end2] = interval2;

    // 有交集的条件：一个区间的开始小于另一个区间的结束，且另一个区间的开始小于这个区间的结束
    return Math.max(start1, start2) < Math.min(end1, end2);
  }

  _mergeIntervals(intervals) {
    if (!intervals || intervals.length === 0) {
      return [];
    }

    // 按起始位置排序
    const sortedIntervals = intervals.sort((a, b) => a[0] - b[0]);
    const merged = [];

    for (const interval of sortedIntervals) {
      if (merged.length === 0) {
        merged.push([...interval]);
      } else {
        const [lastStart, lastEnd] = merged[merged.length - 1];
        const [currentStart, currentEnd] = interval;

        // 如果当前区间与上一个区间重叠或相邻
        if (currentStart <= lastEnd) {
          // 合并区间，取最大的结束位置
          merged[merged.length - 1][1] = Math.max(lastEnd, currentEnd);
        } else {
          merged.push([...interval]);
        }
      }
    }

    return merged.map(interval => [...interval]);
  }

  _mergeFreeIntervals() {
    this.free = this._mergeIntervals(this.free);
  }

  getOccupied() {
    return this.occupied.map(interval => [...interval]);
  }

  getFree() {
    return this.free.map(interval => [...interval]);
  }

  getAllIntervals() {
    return {
      occupied: this.getOccupied(),
      free: this.getFree()
    };
  }

  clear() {
    this.occupied = [];
    this.free = [];
  }

  getUsedPercentage() {
    const totalOccupied = this.occupied.reduce((sum, [start, end]) => sum + (end - start), 0);
    return totalOccupied / this.max;
  }

  toString() {
    const currentNeuron = this.occupied.reduce((sum, [start, end]) => sum + (end - start), 0);
    return `占用百分比: ${(currentNeuron / this.max).toFixed(4)}, ${currentNeuron}/${this.max}`;
  }
}

//单步区间迭代函数
function getStates(infos) {
  const neuronAddr = infos.neuron_addr;
  const neuronSize = infos.neuron_size;
  // const locs = [];
  const utils = [];
  const intervals = [];

  // 初始化区间管理器
  const manager = new IntervalManager(neuronSize);
  for (const op of infos.info) {
    const op_name = op.op.split("_")[0]
    for (const info of op.output_info) {
      const start = info.start;
      if (start < 0) {
        // address of weights, pass
        continue;
      }
      const end = info.end;
      manager.addIntervals([[start, end, op_name+"_"+info.locs]]);
    }
    utils.push(manager.getUsedPercentage());
    intervals.push(manager.getOccupied());
  }
  return {
    utils: utils,
    intervals: intervals,
    neuron_size: neuronSize
  };
}

function openFilePicker() {
  // ✅ 安全检查：确保 ref 存在且元素已挂载
  const file_picker= document.getElementById('fileInput');
  file_picker.click();
}

function trimDictByStartAddr(data, threshold = 3) {
  const keys = Object.keys(data);
  if (keys.length === 0) return {};

  // 1. 提取所有 startAddr 值
  const addrs = keys.map(key => data[key].startAddr);
  console.log('addrs', addrs);

  // 2. 计算中位数
  const sortedAddrs = [...addrs].sort((a, b) => a - b);
  const median = sortedAddrs[Math.floor(sortedAddrs.length / 2)];
  console.log('median', median);

  // 3. 计算绝对偏差
  const deviations = addrs.map(addr => Math.abs(addr - median));
  console.log('deviations', deviations);

  // 4. 计算 MAD
  const madSorted = [...deviations].sort((a, b) => a - b);
  const mad = madSorted[Math.floor(madSorted.length / 2)];

  // 5. 处理 MAD 为 0 的情况（所有值相同）
  if (mad === 0) {
    const normalKeys = keys.filter(key => data[key].startAddr === median);
    return Object.fromEntries(normalKeys.map(k => [k, data[k]]));
  }

  // 6. 过滤：保留非离群点
  const keptEntries = keys
    .filter(key => {
      const modZScore = 0.6745 * (data[key].startAddr - median) / mad;
      return Math.abs(modZScore) < threshold;
    })
    .map(key => [key, data[key]]);

  return Object.fromEntries(keptEntries);
}

function trimObjectsByStartAddr(data, threshold = 3) {
  if (data.length === 0) return [];

  // 1. 提取所有 startAddr 值
  const addrs = data.map(item => item.start);

  // 2. 计算中位数
  const sortedAddrs = [...addrs].sort((a, b) => a - b);
  const median = sortedAddrs[Math.floor(sortedAddrs.length / 2)];

  // 3. 计算绝对偏差
  const deviations = addrs.map(addr => Math.abs(addr - median));

  // 4. 计算 MAD（绝对中位差）
  const madSorted = [...deviations].sort((a, b) => a - b);
  const mad = madSorted[Math.floor(madSorted.length / 2)];

  // 5. 处理 MAD 为 0 的情况（所有 start 相同）
  if (mad === 0) {
    return data.filter(item => item.start === median);
  }

  // 6. 使用 Modified Z-Score 过滤对象
  return data.filter(item => {
    const modZScore = 0.6745 * (item.start - median) / mad;
    return Math.abs(modZScore) < threshold;
  });
}

function merge_ts_address(log, final_mlir_info) {
    // //筛选，如果log里的startAddr>128M,丢掉
    // function madOutlierRemoval(data, threshold = 3) {
    //   if (data.length === 0) return [];

    //   // 1. 计算中位数
    //   const sorted = [...data].sort((a, b) => a - b);
    //   const median = sorted[Math.floor(sorted.length / 2)];

    //   // 2. 计算每个点到中位数的绝对偏差
    //   const deviations = data.map(x => Math.abs(x - median));

    //   // 3. 计算 MAD（绝对中位差）
    //   const madSorted = [...deviations].sort((a, b) => a - b);
    //   const mad = madSorted[Math.floor(madSorted.length / 2)];

    //   // 4. 如果 MAD 为 0（所有值相同），直接返回原数组或去重
    //   if (mad === 0) {
    //     return data.filter(x => x === median);
    //   }

    //   // 5. 计算 modified Z-score，并过滤
    //   return data.filter(x => {
    //     const modZScore = 0.6745 * (x - median) / mad;
    //     return Math.abs(modZScore) < threshold;
    //   });
    // }

    // 计算最大时间戳
    const maxTs = Math.max(...Object.values(log)
        .filter(x => x.live_end < 1e6)
        .map(x => x.live_end)
    );
    const res = []
    // 遍历final_mlir_info中的信息并更新时间戳
    for (const x of final_mlir_info.info) {
        for (const o of x.output_info) {
            if (o.start >= 0) {
                const locationKey = mapping_locs_name.value[o.locs];

                if (!log.hasOwnProperty(locationKey)) {
                    console.log("inplace op:    " + locationKey + " locs: "+ o.locs);
                    continue;
                }

                o.start_ts = log[locationKey].live_start;
                o.end_ts = log[locationKey].live_end;

                if (o.end_ts > maxTs) {
                    o.end_ts = maxTs;
                }
                res.push({"name": o.locs, "start": o.start_ts, "end": o.end_ts, "startAddr": o.start, "size": o.end-o.start})
            }
        }
    }
    // console.log(chip.value)
    console.log("here")
    console.log(iscv181x.value)
    if (iscv181x.value) {
      console.log("cv181x")
      console.log("input_nums: ",input_nums.value)
      console.log("output_nums: ",output_nums.value)
      return res.slice(input_nums.value,-output_nums.value);
    }
    return res;
}

function loadLOGFromFileInput(fileInput) {
  return new Promise((resolve, reject) => {
    if (!fileInput.files || fileInput.files.length === 0) {
      reject(new Error('请选择文件'));
      return;
    }

    const file = fileInput.files[0];
    const reader = new FileReader();

    reader.onload = function(e) {
      const final_mlir_info = parseFinalMlir_allsubnet(e.target.result);
      const data = getStates(final_mlir_info);
      gantt_log.value = merge_ts_address(transform_ts_log(), final_mlir_info);
      // console.log("index_name: ")
      // console.log(mapping_index_name.value);
      // console.log("loc_name: ")
      // console.log(mapping_locs_name.value)
      console.log("final_mlir_info: ")
      console.log(final_mlir_info);
      console.log("transform_ts_log")
      console.log(transform_ts_log());
      console.log("gantt_log: ")
      console.log(gantt_log.value);
      // console.log("subnet_ios.value: ")
      // console.log(subnet_ios.value)

      colors_gantt.value = [];
      for (let i = 0; i < gantt_log.value.length + 1; i++) {
        const hue = (i * 137.508) % 360;
        colors_gantt.value.push(`hsl(${hue}, 70%, 60%)`);
      }
      resolve(data);
    };

    reader.onerror = function() {
      reject(new Error('文件读取失败'));
    };

    reader.readAsText(file);
  });
}

// 处理文件选择
const handleFileChange = async (event) => {
  const file = event.target.files[0];
  if (!file) return

  try {
    mapping_locs_name.value = {};
    mapping_index_name.value = {};
    mapping_tensor_subnetID.value = {};
    gantt_log.value = {};
    subnet_ios.value = [];
    const data = await loadLOGFromFileInput(event.target)
    draw(data)
    drawGanttChart(true);

    const canvas = document.getElementById('ganttChart');
    canvas.addEventListener('wheel', handleWheel, { passive: false });
    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mousemove', handleGanttMouseMove);
    canvas.addEventListener('mouseup', handleMouseUp);
    canvas.addEventListener('mouseleave', handleMouseUp);

    // //检查柱状图，展示信息
    // canvas.addEventListener('mousemove', (event) => {
    //   if (isDragging.value) return;
    //   const rect = canvas.getBoundingClientRect();
    //   const mouseX = event.clientX - rect.left;
    //   const mouseY = event.clientY - rect.top;
    //   const bars = JSON.parse(canvas.dataset.bars || '[]');
    //   for (const bar of bars) {
    //       if (
    //           mouseX >= bar.x &&
    //           mouseX <= bar.x + bar.width &&
    //           mouseY >= bar.y &&
    //           mouseY <= bar.y + bar.height
    //       ) {
    //           showMemInfo(event, bar.index);
    //           return;
    //       }
    //   }
    // });

    // // 滑块事件监听
    // document.getElementById('timelineSlider').addEventListener('input', handleSliderChange);
    } catch (error) {
        console.error('加载文件失败:', error)
    }
}

// 绘制主柱状图
function drawMainChart() {
    const canvas = document.getElementById('mainChart');
    const ctx = canvas.getContext('2d');
    canvas.width = document.getElementById('chart-container').getBoundingClientRect().width;
    const width = canvas.width;
    const height = canvas.height;


    // 清空画布
    ctx.clearRect(0, 0, width, height);

    // 设置边距
    const margin = { top: 30, right: 30, bottom: 60, left: 60 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    // 绘制坐标轴
    ctx.beginPath();
    ctx.moveTo(margin.left, height - margin.bottom);
    ctx.lineTo(width - margin.right, height - margin.bottom); // X轴
    ctx.lineTo(width - margin.right, margin.top); // Y轴
    ctx.strokeStyle = '#333';
    ctx.stroke();

    // 绘制Y轴标签
    ctx.fillStyle = '#333';
    ctx.font = '14px Arial';
    ctx.textAlign = 'right';
    for (let y = 0; y <= 1; y += 0.2) {
        const yPos = height - margin.bottom - (y * chartHeight);
        ctx.fillText(y.toFixed(1), margin.left - 10, yPos + 4);

        // 绘制网格线
        ctx.beginPath();
        ctx.moveTo(margin.left, yPos);
        ctx.lineTo(width - margin.right, yPos);
        ctx.strokeStyle = '#eee';
        ctx.stroke();
    }

    // 绘制X轴标签和柱子
    const barCount = utils.value.length;
    const barWidth = chartWidth / barCount * 0.8;
    const spacing = chartWidth / barCount * 0.2;

    for (let i = 0; i < barCount; i++) {
        const x = margin.left + i * (barWidth + spacing) + spacing / 2;
        const barHeight = utils.value[i] * chartHeight;
        const y = height - margin.bottom - barHeight;

        // 绘制柱子
        ctx.fillStyle = '#4a90e2';
        ctx.fillRect(x, y, barWidth, barHeight);

        // 绘制柱子边框
        ctx.strokeStyle = '#357abd';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, barWidth, barHeight);

        // 绘制X轴标签
        ctx.textAlign = 'center';

        // 添加交互区域信息（存储在数据属性中）
        if (!canvas.dataset.bars) canvas.dataset.bars = '[]';
        const bars = JSON.parse(canvas.dataset.bars || '[]');
        bars.push({
            index: i,
            x: x,
            y: y,
            width: barWidth,
            height: barHeight
        });
        canvas.dataset.bars = JSON.stringify(bars);
    }

    // 绘制Y轴标签
    // ctx.textAlign = 'right';
    // ctx.fillText('空间利用率', 20, margin.top / 2);

    // // 绘制标题
    // ctx.textAlign = 'center';
    // ctx.font = 'bold 16px Arial';
    // ctx.fillText('时间步空间利用率', width / 2, 20);
}

// 绘制内存分配条形图
function drawMemoryChart(plotData, canvasId, timeStepIndex) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // 清空画布
    ctx.clearRect(0, 0, width, height);

    // 设置边距
    const margin = { top: 20, right: 20, bottom: 40, left: 60 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;


    // 绘制每个内存段
    const barY = margin.top + chartHeight / 4;
    const barHeight = chartHeight / 2;

    for (const segment of plotData) {
        const startAddr = segment[0];
        const endAddr = segment[1];
        const segmentId = Segments.value.indexOf(String(segment[0])+"_"+String(segment[1]));

        const x = margin.left + (startAddr / neuron_size.value) * chartWidth;
        const w = ((endAddr - startAddr) / neuron_size.value) * chartWidth;

        // 绘制内存段
        ctx.fillStyle = colors.value[segmentId % colors.value.length];
        ctx.fillRect(x, barY, w, barHeight);

        // 绘制边框
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, barY, w, barHeight);

        // 在段中间绘制标签
        if (w > 40) { // 只有当段足够宽时才显示标签
            ctx.fillStyle = '#fff';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(
                // segment[2],
                segment[2].split("#")[1],
                x + w / 2,
                barY + barHeight / 2 + 4
            );
        }
    }

    // 绘制X轴
    ctx.beginPath();
    ctx.moveTo(margin.left, barY + barHeight + 10);
    ctx.lineTo(width - margin.right, barY + barHeight + 10);
    ctx.strokeStyle = '#333';
    ctx.stroke();

    // 绘制X轴标签
    ctx.fillStyle = '#333';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';

    // 绘制几个关键点的标签
    for (let i = 0; i <= 5; i++) {
        const addrValue = (neuron_size.value / 5) * i;
        const xPos = margin.left + (addrValue / neuron_size.value) * chartWidth;
        ctx.fillText(
            `${Math.round(addrValue / 1000000)}M`,
            xPos,
            barY + barHeight + 30
        );

        // 绘制刻度线
        ctx.beginPath();
        ctx.moveTo(xPos, barY + barHeight + 8);
        ctx.lineTo(xPos, barY + barHeight + 12);
        ctx.stroke();
    }

    // 绘制标题
    ctx.textAlign = 'center';
    ctx.font = 'bold 14px Arial';
    ctx.fillText('内存分配详情', width / 2, 15);
}

// 鼠标跟随显示工具提示
function showTooltip(event, timeStepIndex) {
    const tooltip = document.getElementById('tooltip');
    tooltip.innerHTML = `
        <div style="font-weight: bold; margin-bottom: 10px;">时间步 T${timeStepIndex + 1}</div>
        <div>空间利用率: ${(utils.value[timeStepIndex] * 100).toFixed(2)}%</div>
        <div style="margin-top: 10px; font-weight: bold;">内存分配详情:</div>
        <canvas id="memoryChart${timeStepIndex}" width="800" height="150"></canvas>
        <div style="margin-top: 5px; font-size: 12px;">
            ${intervals.value[timeStepIndex].map(s =>
                `地址: ${s[0]}-${s[1]} (${s[2]})`
            ).join('<br>')}
        </div>
    `;
    tooltip.style.display = 'block';

    // 更新位置
    updateTooltipPosition(event);

    // 延迟绘制内存图表，确保DOM元素已渲染
    setTimeout(() => {
        drawMemoryChart(intervals.value[timeStepIndex], `memoryChart${timeStepIndex}`,timeStepIndex);
    }, 10);
}

// 更新工具提示位置
function updateTooltipPosition(event) {
    const tooltip = document.getElementById('tooltip');
    if (tooltip.style.display === 'none') return;

    let x = event.pageX + 15;
    let y = event.pageY - tooltip.offsetHeight - 15;

    // 检查是否超出右边界
    if (x + tooltip.offsetWidth > window.innerWidth) {
        x = event.pageX - tooltip.offsetWidth - 15;
    }

    // 检查是否超出上边界
    if (y < 0) {
        y = event.pageY + 15;
    }

    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
}

// 隐藏工具提示
function hideTooltip() {
    const tooltip = document.getElementById('tooltip');
    tooltip.style.display = 'none';
}

// 初始化
function draw(data){
  utils.value = data.utils
  intervals.value = data.intervals;
  neuron_size.value = data.neuron_size;
  drawMainChart();

  // 获取所有内存段的总数量以确定颜色数量
  let maxSegmentId = 0;
  Segments.value = new Set();
  for (const interval of intervals.value) {
    for (const segment of interval) {
      Segments.value.add(String(segment[0])+"_"+String(segment[1]));
    }
  }
  Segments.value = Array.from(Segments.value);
  for (let i = 0; i < Segments.value.length + 1; i++) {
    // 使用HSL颜色模型生成不同色调的颜色
    const hue = (i * 137.508) % 360; // 黄金角度，确保颜色分布均匀
    colors.value.push(`hsl(${hue}, 70%, 60%)`);
  }

  const canvas = document.getElementById('mainChart');

  // // 添加鼠标移动事件监听器
  // canvas.addEventListener('mousemove', (event) => {
  //     const rect = canvas.getBoundingClientRect();
  //     const mouseX = event.clientX - rect.left;
  //     const mouseY = event.clientY - rect.top;

  //     // 解析柱子数据
  //     const bars = JSON.parse(canvas.dataset.bars || '[]');

  //     // 检查鼠标是否在任一柱子上
  //     for (const bar of bars) {
  //         if (
  //             mouseX >= bar.x &&
  //             mouseX <= bar.x + bar.width &&
  //             mouseY >= bar.y &&
  //             mouseY <= bar.y + bar.height
  //         ) {
  //             canvas.style.cursor = 'pointer';
  //             return;
  //         }
  //     }

  //     canvas.style.cursor = 'default';
  // });

  // 添加点击事件监听器
  canvas.addEventListener('mousemove', (event) => {
      const rect = canvas.getBoundingClientRect();
      const mouseX = event.clientX - rect.left;
      const mouseY = event.clientY - rect.top;

      // 解析柱子数据
      const bars = JSON.parse(canvas.dataset.bars || '[]');
      // 检查点击的是哪个柱子
      for (const bar of bars) {
          if (
              mouseX >= bar.x &&
              mouseX <= bar.x + bar.width &&
              mouseY >= bar.y &&
              mouseY <= bar.y + bar.height
          ) {
              showTooltip(event, bar.index);
              return;
          }
      }
  });

  // 添加鼠标移出事件监听器
  canvas.addEventListener('mouseout', () => {
      canvas.style.cursor = 'default';
  });

  // // 工具提示跟随鼠标
  // document.addEventListener('mousemove', (event) => {
  //     updateTooltipPosition(event);
  // });

  // 点击页面其他地方隐藏工具提示
  document.addEventListener('click', (event) => {
      const tooltip = document.getElementById('tooltip');
      const canvas = document.getElementById('mainChart');

      if (
          !tooltip.contains(event.target) &&
          !canvas.contains(event.target)
      ) {
          hideTooltip();
      }
  });
};

function showMemInfo(event, timeStepIndex) {
    const data = gantt_log.value[timeStepIndex];
    const tooltip = document.getElementById('tooltip-gantt');
    tooltip.innerHTML = `
        <div style="font-weight: bold; margin-bottom: 10px;">时间步 T${timeStepIndex + 1}</div>
        <div style="margin-top: 5px; font-size: 12px;">
            起始地址: ${data.startAddr}<br>
            占用大小: ${data.size}<br>
            起始时间步: ${data.start}<br>
            结束时间步: ${data.end}<br>
            进程: ${data.name}
        </div>
    `;
    tooltip.style.display = 'block';

    // 更新位置
    if (tooltip.style.display === 'none') return;

    let x = event.pageX + 15;
    let y = event.pageY - tooltip.offsetHeight - 15;


    if (x + tooltip.offsetWidth > window.innerWidth) {// 检查是否超出右边界
        x = event.pageX - tooltip.offsetWidth - 15;
    }


    if (y < 0) {// 检查是否超出上边界
        y = event.pageY + 15;
    }

    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
}

// 绘制甘特图
function drawGanttChart(init = false) {
    const canvas = document.getElementById('ganttChart');
    const ctx = canvas.getContext('2d');
    canvas.width = document.getElementById('chart-container-gantt').getBoundingClientRect().width;
    // 清空画布
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 计算时间轴和内存地址范围
    let timeMin = Infinity, timeMax = -Infinity;
    let addrMin = Infinity, addrMax = -Infinity;
    gantt_log.value.forEach(proc => {
        timeMin = Math.min(timeMin, proc.start, proc.end);
        timeMax = Math.max(timeMax, proc.start, proc.end);
        addrMin = Math.min(addrMin, proc.startAddr);
        addrMax = Math.max(addrMax, proc.startAddr + proc.size);
    });

    // 添加边距
    // const timeMargin = (timeMax - timeMin) * 0.05 || 1;
    // const addrMargin = (addrMax - addrMin) * 0.05 || 10;
    const timeMargin = 0;
    const addrMargin = 0;

    timeMin -= timeMargin;
    timeMax += timeMargin;
    addrMin -= addrMargin;
    addrMax += addrMargin;

    if (init){
      viewEndTime.value = timeMax;
      fullTimeRange.value.max = timeMax;
    }

    // 设置画布绘图区域
    const margin = { top: 50, right: 50, bottom: 30, left: 100 };
    const chartWidth = canvas.width - margin.left - margin.right;
    const chartHeight = canvas.height - margin.top - margin.bottom;

    // 计算当前视图的时间范围
    const currentTimeMin = init ? timeMin : viewStartTime.value ;
    const currentTimeMax = init ? timeMax: viewEndTime.value;

    // 绘制背景网格
    ctx.strokeStyle = '#eee';
    ctx.lineWidth = 1;

    // 绘制x轴网格线（时间轴）
    const timeTicks = generateTicks(currentTimeMin, currentTimeMax, 10);
    timeTicks.forEach((time, index) => {
        const x = margin.left + ((time - currentTimeMin) / (currentTimeMax - currentTimeMin)) * chartWidth;
        ctx.beginPath();
        ctx.moveTo(x, margin.top);
        ctx.lineTo(x, margin.top + chartHeight);
        ctx.stroke();

        // 时间轴标签
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(Math.round(time), x, canvas.height - 10);
    });

    // 绘制y轴网格线（内存地址轴）
    const addrTicks = generateTicks(addrMin, addrMax, 10);
    addrTicks.forEach((addr, index) => {
        const y = margin.top + chartHeight - ((addr - addrMin) / (addrMax - addrMin)) * chartHeight;
        ctx.beginPath();
        ctx.moveTo(margin.left, y);
        ctx.lineTo(margin.left + chartWidth, y);
        ctx.stroke();

        // 内存地址轴标签
        ctx.fillStyle = '#333';
        ctx.font = '12px Arial';
        ctx.textAlign = 'right';
        ctx.fillText(addr.toFixed(0), margin.left - 10, y + 4);
    });

    // 绘制坐标轴
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 2;

    // X轴
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top + chartHeight);
    ctx.lineTo(margin.left + chartWidth, margin.top + chartHeight);
    ctx.stroke();

    // Y轴
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top);
    ctx.lineTo(margin.left, margin.top + chartHeight);
    ctx.stroke();

    // Y轴标签
    ctx.save();
    ctx.translate(20, canvas.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Memory Address Space', 0, 0);
    ctx.restore();

    // 清除之前存储的交互区域信息
    canvas.dataset.bars = '[]';

    // 绘制每个进程的内存分配矩形
    gantt_log.value.forEach((proc, index) => {
        // 只绘制在当前时间范围内的进程
        if (proc.start <= currentTimeMax && proc.end >= currentTimeMin) {
            const visibleStart = Math.max(proc.start, currentTimeMin);
            const visibleEnd = Math.min(proc.end, currentTimeMax);

            const x = margin.left + ((visibleStart - currentTimeMin) / (currentTimeMax - currentTimeMin)) * chartWidth;
            const y = margin.top + chartHeight - ((proc.startAddr + proc.size - addrMin) / (addrMax - addrMin)) * chartHeight;
            const width = ((visibleEnd - visibleStart) / (currentTimeMax - currentTimeMin)) * chartWidth;
            const height = (proc.size / (addrMax - addrMin)) * chartHeight;

            // 绘制矩形
            ctx.fillStyle = colors_gantt.value[index];
            ctx.fillRect(x, y, width, height);

            // 绘制边框
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 1;
            ctx.strokeRect(x, y, width, height);

            // 绘制文字标注
            ctx.fillStyle = '#000';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';

            if (height > 20 && width > 50) {
                ctx.fillText(proc.name, x + width/2, y + height/2 + 5);
                // ctx.fillText(`${proc.startAddr}-${proc.startAddr + proc.size}`, x + width/2, y + height/2 + 20);
            } else {
                ctx.fillText(proc.name, x + width/2, y - 10);
            }

            // 存储交互区域信息
            const bars = JSON.parse(canvas.dataset.bars || '[]');
            bars.push({
                index: index,
                x: x,
                y: y,
                width: width,
                height: height
            });
            canvas.dataset.bars = JSON.stringify(bars);
        }
    // 添加点击事件监听器
    });

}

// 生成刻度值
function generateTicks(min, max, count) {
    const range = max - min;
    const step = range / count;
    const ticks = [];
    for (let i = 0; i <= count; i++) {
        ticks.push(min + i * step);
    }
    return ticks;
}

// 存储缩放和偏移状态
const isDragging = ref(false);
const startX = ref(0);
const startY = ref(0);
const viewStartTime = ref(0); // 当前视图开始时间
const viewEndTime = ref(15);  // 当前视图结束时间
const fullTimeRange = ref({ min: 0, max: 15 }); // 完整时间范围


// // 更新时间轴滑块
// function updateTimelineSlider() {
//     const slider = document.getElementById('timelineSlider');
//     const labelsContainer = document.getElementById('timeLabels');

//     // 设置滑块值
//     const totalRange = fullTimeRange.value.max - fullTimeRange.value.min;
//     const currentStartPos = ((viewStartTime.value - fullTimeRange.value.min) / totalRange) * 100;
//     const currentEndPos = ((viewEndTime.value - fullTimeRange.value.min) / totalRange) * 100;

//     // 这里我们只设置一个值，代表当前视图中心
//     const centerTime = (viewStartTime.value + viewEndTime.value) / 2;
//     const centerPos = ((centerTime - fullTimeRange.value.min) / totalRange) * 100;
//     slider.value = centerPos;

//     // 更新时间标签
//     labelsContainer.innerHTML = `${fullTimeRange.value.min.toFixed(1)} - ${fullTimeRange.value.max.toFixed(1)}`;
// }

// 缩放控制函数
function zoomIn() {
    const zoomFactor = 0.8;
    const timeRange = viewEndTime.value - viewStartTime.value;
    const newTimeRange = timeRange * zoomFactor;
    const centerTime = (viewStartTime.value + viewEndTime.value) / 2;

    viewStartTime.value = centerTime - newTimeRange / 2;
    viewEndTime.value = centerTime + newTimeRange / 2;

    // 限制在完整时间范围内
    if (viewStartTime.value < fullTimeRange.value.min) {
        const diff = fullTimeRange.value.min - viewStartTime.value;
        viewStartTime.value = fullTimeRange.value.min;
        viewEndTime.value += diff;
    }
    if (viewEndTime.value > fullTimeRange.value.max) {
        const diff = viewEndTime.value - fullTimeRange.value.max;
        viewEndTime.value = fullTimeRange.value.max;
        viewStartTime.value -= diff;
    }

    // updateTimelineSlider();
    drawGanttChart();
}

function zoomOut() {
    const zoomFactor = 1.25;
    const timeRange = viewEndTime.value - viewStartTime.value;
    const newTimeRange = timeRange * zoomFactor;
    const centerTime = (viewStartTime.value + viewEndTime.value) / 2;

    viewStartTime.value = centerTime - newTimeRange / 2;
    viewEndTime.value = centerTime + newTimeRange / 2;

    // // 限制在完整时间范围内
    // if (viewStartTime.value < fullTimeRange.value.min) {
    //     viewStartTime.value = fullTimeRange.value.min;
    // }
    // if (viewEndTime.value > fullTimeRange.value.max) {
    //     viewEndTime.value = fullTimeRange.value.max;
    // }
    // updateTimelineSlider();
    drawGanttChart();
}

function resetView() {
    viewStartTime.value = fullTimeRange.value.min;
    viewEndTime.value = fullTimeRange.value.max;
    // updateTimelineSlider();
    drawGanttChart(true);
}

// // 滑块改变事件
// function handleSliderChange() {
//     const slider = document.getElementById('timelineSlider');
//     const sliderValue = parseInt(slider.value);

//     // 计算当前视图的时间范围
//     const totalRange = fullTimeRange.value.max - fullTimeRange.value.min;
//     const viewRange = viewEndTime.value - viewStartTime.value;
//     const centerPos = (sliderValue / 100) * totalRange;

//     // 以滑块位置为中心，调整视图范围
//     viewStartTime.value = centerPos - viewRange / 2;
//     viewEndTime.value = centerPos + viewRange / 2;

//     // 限制在完整时间范围内
//     if (viewStartTime.value < fullTimeRange.value.min) {
//         const diff = fullTimeRange.value.min - viewStartTime;
//         viewStartTime.value = fullTimeRange.value.min;
//         viewEndTime.value += diff;
//     }
//     if (viewEndTime.value > fullTimeRange.value.max) {
//         const diff = viewEndTime.value - fullTimeRange.value.max;
//         viewEndTime.value = fullTimeRange.value.max;
//         viewStartTime.value -= diff;
//     }

//     drawGanttChart();
// }

// 鼠标滚轮缩放
function handleWheel(event) {
    const canvas = document.getElementById('ganttChart');
    event.preventDefault();

    const wheel = event.deltaY < 0 ? 1 : -1;
    const zoomIntensity = 0.1;
    const factor = wheel > 0 ? (1 - zoomIntensity) : (1 + zoomIntensity);

    // 计算当前鼠标位置对应的时间
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const margin = { left: 100 };
    const chartWidth = canvas.width - margin.left - 50;
    const mouseTime = viewStartTime.value + ((mouseX - margin.left) / chartWidth) * (viewEndTime.value - viewStartTime.value);

    // 确保chartWidth是正数，防止除零错误
    if (chartWidth <= 0) return;

    // 调整时间范围
    const oldRange = viewEndTime.value - viewStartTime.value;
    const newRange = oldRange * factor;
    const newStartTime = mouseTime - ((mouseTime - viewStartTime.value) / oldRange) * newRange;
    const newEndTime = newStartTime + newRange;

    // 限制在完整时间范围内
    if (newStartTime < fullTimeRange.value.min) {
        viewStartTime.value = fullTimeRange.value.min;
        viewEndTime.value = viewStartTime.value + newRange;
    } else if (newEndTime > fullTimeRange.value.max) {
        viewEndTime.value = fullTimeRange.value.max;
        viewStartTime.value = viewEndTime.value - newRange;
    } else {
        viewStartTime.value = newStartTime;
        viewEndTime.value = newEndTime;
    }
    // 确保时间范围不小于最小值
    const minRange = 0.1; // 最小时间范围
    if (viewEndTime - viewStartTime < minRange) {
        const center = (viewStartTime + viewEndTime) / 2;
        viewStartTime = center - minRange / 2;
        viewEndTime = center + minRange / 2;

        // 再次检查边界
        if (viewStartTime < fullTimeRange.min) {
            viewStartTime = fullTimeRange.min;
            viewEndTime = viewStartTime + minRange;
        }
        if (viewEndTime > fullTimeRange.max) {
            viewEndTime = fullTimeRange.max;
            viewStartTime = viewEndTime - minRange;
        }
    }

    // updateTimelineSlider();
    drawGanttChart();
}

// 拖拽平移
function handleMouseDown(event) {
    const canvas = document.getElementById('ganttChart');
    isDragging.value = true;
    startX.value = event.clientX;
    startY.value = event.clientY;
    canvas.style.cursor = 'grabbing';
}

function handleGanttDragdingMouseMove(event) {
    const canvas = document.getElementById('ganttChart');
    if (!isDragging.value) return;

    const deltaX = event.clientX - startX.value;
    const margin = { left: 100 };
    const chartWidth = canvas.width - margin.left - 50;
    const timePerPixel = (viewEndTime.value - viewStartTime.value) / chartWidth;
    const timeDelta = -deltaX * timePerPixel;

    viewStartTime.value += timeDelta;
    viewEndTime.value += timeDelta;

    // 限制在完整时间范围内
    if (viewStartTime.value < fullTimeRange.value.min) {
        const diff = fullTimeRange.value.min - viewStartTime.value;
        viewStartTime.value = fullTimeRange.value.min;
        viewEndTime.value += diff;
    }
    if (viewEndTime.value > fullTimeRange.value.max) {
        const diff = viewEndTime.value - fullTimeRange.value.max;
        viewEndTime.value = fullTimeRange.value.max;
        viewStartTime.value -= diff;
    }

    startX.value = event.clientX;
    startY.value = event.clientY;

    // updateTimelineSlider();
    drawGanttChart();
}

// 鼠标移动事件处理（优化性能）
let hoverTimeout = null;
function handleGanttMouseMove(event) {
    const canvas = document.getElementById('ganttChart');
    if (isDragging.value){ //拖拽事件
      const deltaX = event.clientX - startX.value;
      const margin = { left: 100 };
      const chartWidth = canvas.width - margin.left - 50;
      const timePerPixel = (viewEndTime.value - viewStartTime.value) / chartWidth;
      const timeDelta = -deltaX * timePerPixel;

      viewStartTime.value += timeDelta;
      viewEndTime.value += timeDelta;

      // 限制在完整时间范围内
      if (viewStartTime.value < fullTimeRange.value.min) {
          const diff = fullTimeRange.value.min - viewStartTime.value;
          viewStartTime.value = fullTimeRange.value.min;
          viewEndTime.value += diff;
      }
      if (viewEndTime.value > fullTimeRange.value.max) {
          const diff = viewEndTime.value - fullTimeRange.value.max;
          viewEndTime.value = fullTimeRange.value.max;
          viewStartTime.value -= diff;
      }

      startX.value = event.clientX;
      startY.value = event.clientY;

      // updateTimelineSlider();
      drawGanttChart();
    } else { //内存展示事件
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      // 使用防抖优化频繁的鼠标移动事件
      clearTimeout(hoverTimeout);
      hoverTimeout = setTimeout(() => {
          const bars = JSON.parse(canvas.dataset.bars || '[]');

          for (let i = 0; i < bars.length; i++) {
              const bar = bars[i];
              if (x >= bar.x && x <= bar.x + bar.width &&
                  y >= bar.y && y <= bar.y + bar.height) {

                  // // 高亮显示矩形
                  // const ctx = canvas.getContext('2d');
                  // ctx.strokeStyle = '#ff0000';
                  // ctx.lineWidth = 3;
                  // ctx.strokeRect(bar.x, bar.y, bar.width, bar.height);

                  // 显示提示框
                  showMemInfo(event, bar.index);
                  return;
              }
          }

          // 如果没有找到匹配的矩形，隐藏提示框
          const tooltip = document.getElementById('tooltip-gantt');
          tooltip.style.display = 'none';
      }, 10); // 10ms防抖延迟
    }
}

// 鼠标离开事件
function handleGanttMouseLeave() {
    const canvas = document.getElementById('ganttChart');
    // 如果鼠标离开画布，取消高亮
    if (highlightedBar) {
        highlightedBar = null;
        drawGanttChart(); // 重新绘制整个图表
    }
    hideTooltip();
}

function handleMouseUp() {
    const canvas = document.getElementById('ganttChart');
    isDragging.value = false;
    canvas.style.cursor = 'default';
}

// // 更新图表
// function updateChart() {
//     drawGanttChart();
// }

</script>

<style scoped>
body {
          background-color: #f0f0f0;
          font-family: Arial, sans-serif;
          margin: 0;
          padding: 20px;
      }

      .container {
          max-width: 90%;
          margin: 0 auto;
      }

      h1 {
          text-align: center;
          color: #333;
      }
      #chart-container {
          width: 100%;
          height: 100%;
          background-color: #ffffff;
          /* padding: 20px; */
          /* border-radius: 8px; */
          /* box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); */
          /* margin-bottom: 20px; */
      }
      #chart-container {
          width: 100%;
          height: 100%;
          background-color: #ffffff;
          /* padding: 20px; */
          /* border-radius: 8px; */
          /* box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); */
          /* margin-bottom: 20px; */
      }

      canvas {
          display: block;
          margin: 0 auto;
      }

      #tooltip {
          position: absolute;
          background-color: #ffffff;
          border: 1px solid #cccccc;
          padding: 10px;
          border-radius: 4px;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
          display: none;
          z-index: 1000;
          font-size: 14px;
          color: #000000;
      }

      #tooltip-gantt {
          position: absolute;
          background-color: #ffffff;
          border: 1px solid #cccccc;
          padding: 10px;
          border-radius: 4px;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
          display: none;
          z-index: 1000;
          font-size: 14px;
          color: #000000;
      }

      .legend {
          display: flex;
          justify-content: center;
          margin-top: 10px;
          flex-wrap: wrap;
      }

      .legend-item {
          display: flex;
          align-items: center;
          margin: 0 10px;
      }

      .legend-color {
          width: 20px;
          height: 20px;
          margin-right: 5px;
          border: 1px solid #ccc;
      }
.new-page {
  padding: 20px;
}

/* gantt */
.container-gantt {
    max-width: 90%;
    margin: 0 auto;
    background-color: white;
    /* padding: 20px; */
    /* border-radius: 8px; */
    /* box-shadow: 0 2px 10px rgba(0,0,0,0.1); */
}

.legend {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 20px;
    justify-content: center;
}
.legend-item {
    display: flex;
    align-items: center;
    margin-right: 15px;
}
.legend-color {
    width: 20px;
    height: 20px;
    margin-right: 5px;
    border: 1px solid #ccc;
}

.button {
    padding: 8px 16px;
    margin: 0 5px;
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}
.button:hover {
    background-color: #45a049;
}
.info-panel {
    margin-top: 20px;
    padding: 15px;
    background-color: #f9f9f9;
    border-left: 4px solid #4CAF50;
}
.controls {
    text-align: center;
    margin-bottom: 20px;
}

</style>
