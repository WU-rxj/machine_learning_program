<script setup>
import { ref } from 'vue'
import axios from 'axios'

getList()
const value = ref('')
const list = ref([
  { value: '吃饭', isCompleted: true },
  { value: '睡觉', isCompleted: false },
  { value: '打豆豆', isCompleted: false },
])

// 统一使用同一个域名，将 q6zv39 改为 g6zv39
const API_BASE = 'https://g6zv39.laf.run'

async function getList() {
  const res = await axios({
    url: `${API_BASE}/get_list`,
    method: 'GET',
  });
  list.value = res.data.list;
}

async function add() {
  await axios({
    url: `${API_BASE}/add-todo`,  // 统一命名风格 add_todo
    method: 'POST',
    data: {
      value: value.value,
      isCompleted: false
    }
  });
  getList(); // 添加成功后刷新列表
  value.value = ''
}


async function update(id) {
  try {
    const response = await axios({
      url: 'https://g6zv39.laf.run/update_todo',
      method: 'POST',
      data: {
        id,
        completed: !list.value.find(item => item.id === id).completed
      }
    });

    if (response.status === 200) {
      const todo = list.value.find(item => item.id === id);
      if (todo) {
        todo.completed = !todo.completed;
      }
    }
  } catch (error) {
    console.error('更新失败:', error);
    alert('更新待办事项状态失败，请重试');
  }
}

async function del(id) {
  try {
    const response = await axios({
      url: 'https://g6zv39.laf.run/del_todo',
      method: 'POST',
      data: {
        id: id
      }
    });

    if (response.status === 200) {
      // 删除成功后重新获取列表
      getList();
    }
  } catch (error) {
    console.error('删除失败:', error);
    alert('删除待办事项失败，请重试');
  }
}
</script>
<template>
  <div class="todo-app">
    <div class="title">Todo App</div>

    <div class="todo-form">
      <input
        v-model="value"
        type="text"
        class="todo-input"
        placeholder="Add a todo"
      />
      <div @click="add" class="todo-button">Add Todo</div>
    </div>

    <div
      v-for="(item, index) in list"
      :class="[item.isCompleted ? 'completed' : 'item']"
    >
      <div>
        <input @click="update(item._id)" v-model="item.isCompleted" type="checkbox" />
        <span class="name">{{ item.value }}</span>
      </div>

      <div @click="del(index)" class="del">del</div>
    </div>
  </div>
</template>

<style scoped>
.todo-app {
  box-sizing: border-box;
  margin-top: 40px;
  margin-left: 1%;
  padding-top: 30px;
  width: 98%;
  height: 500px;
  background: #ffffff;
  border-radius: 5px;
}

.title {
  text-align: center;
  font-size: 30px;
  font-weight: 700;
}

.todo-form {
  display: flex;
  margin: 20px 0 30px 20px;
}

.todo-button {
  width: 100px;
  height: 52px;
  border-radius: 0 20px 20px 0;

  text-align: center;
  background: linear-gradient(
    to right,
    rgb(113, 65, 168),
    rgba(44, 114, 251, 1)
  );
  color: #fff;
  line-height: 52px;
  cursor: pointer;
  font-size: 14px;
  user-select: none;
}

.todo-input {
  padding: 0px 15px 0px 15px;
  border-radius: 20px 0 0 20px;
  border: 1px solid #dfe1e5;
  outline: none;
  width: 60%;
  height: 50px;
}

.item {
  box-sizing: border-box;
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 80%;
  height: 50px;
  margin: 8px auto;
  padding: 16px;
  border-radius: 20px;
  box-shadow: rgba(149, 157, 165, 0.2) 0px 8px 20px;
}

.del {
  color: red;
}

.completed {
  box-sizing: border-box;
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 80%;
  height: 50px;
  margin: 8px auto;
  padding: 16px;
  border-radius: 20px;
  box-shadow: rgba(149, 157, 165, 0.2) 0px 8px 20px;
  text-decoration: line-through;
  opacity: 0.4;
}
</style>