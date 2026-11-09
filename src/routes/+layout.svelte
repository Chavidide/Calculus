<script lang="ts">
	import './layout.css';
	import favicon from '#lib/assets/favicon.png';
	import type { LayoutProps } from './$types';

	let { children }: LayoutProps = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

<nav class="navbar bg-base-100 top-0 sticky z-50">
	<a href="/" class="btn btn-ghost text-xl" data-sveltekit-reload>수학</a>
</nav>

{@render children()}