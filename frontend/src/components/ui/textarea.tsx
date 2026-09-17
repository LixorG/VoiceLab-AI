import type * as React from 'react'

import { cn } from '@/lib/utils'

function Textarea({ className, ...props }: React.ComponentProps<'textarea'>) {
  return (
    <textarea
      className={cn(
        'min-h-32 w-full resize-y rounded-md border bg-background/50 px-3 py-2 text-sm placeholder:text-muted-foreground focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none disabled:opacity-50',
        className,
      )}
      {...props}
    />
  )
}

export { Textarea }
