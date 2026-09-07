<script setup>
// One accessible popover component serves model, precision, and token help.
defineProps({
  open: {
    type: Boolean,
    default: false,
  },
  label: {
    type: String,
    required: true,
  },
  content: {
    type: Object,
    required: true,
  },
  placement: {
    type: String,
    default: 'settings',
  },
})

defineEmits(['toggle', 'close'])
</script>

<template>
  <span
    class="info-anchor"
    :class="[`info-anchor-${placement}`, { 'info-anchor-open': open }]"
    @click.stop
  >
    <button
      type="button"
      class="info-trigger"
      :aria-label="`Information about ${label}`"
      :aria-expanded="open"
      title="More information"
      @click="$emit('toggle')"
    >
      i
    </button>

    <section v-if="open" class="info-popover" role="dialog" :aria-label="content.title">
      <button
        type="button"
        class="info-close"
        :aria-label="`Close ${content.title} information`"
        @click="$emit('close')"
      >
        ×
      </button>

      <h3>{{ content.title }}</h3>
      <p v-for="paragraph in content.paragraphs" :key="paragraph">{{ paragraph }}</p>

      <ul v-if="content.items">
        <li v-for="item in content.items" :key="item">{{ item }}</li>
      </ul>

      <dl v-if="content.definitions">
        <template v-for="item in content.definitions" :key="item.term">
          <dt>{{ item.term }}</dt>
          <dd>{{ item.description }}</dd>
        </template>
      </dl>

      <p v-if="content.link" class="info-link-line">
        For more details visit:
        <a :href="content.link" target="_blank" rel="noopener noreferrer">Hugging Face</a>
      </p>
    </section>
  </span>
</template>